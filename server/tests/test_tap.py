"""调用抓包（Trace Tap，见 docs/api-traffic-tap.md）的回归测试。

锁定的性质，每条都对应一个「坏了会静默出问题」的点：

  * **脱敏**：Authorization / x-api-key / Cookie 与正文里的 `sk-…`、`Bearer …`
    一律不出现在库里 —— 抓包是全项目唯一存完整对话内容的地方，漏一次就是
    永久留存的密钥；
  * **重建**：流式响应里的工具参数跨帧到达，必须按 index 拼回去（拼不上就是
    一堆半截 JSON）；
  * **会话归并**：同一任务的后续请求落在同一会话、seq 递增，换任务另起一条；
  * **限容**：超长请求体被截断并标记（truncated），不是悄悄丢掉后半段；
  * **旁路安全**：抓包关闭时不写任何行，异常输入也不抛给调用方；
  * **权限**：会话内容只给管理员；只读账号能读开关、读不到对话。
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server import config, db, security  # noqa: E402
from server.services import tap  # noqa: E402

SSE = '\n'.join([
    'data: ' + json.dumps({'choices': [{'delta': {'content': 'Hel'}}]}),
    'data: ' + json.dumps({'choices': [{'delta': {'content': 'lo',
                                                  'reasoning_content': '想一想'}}]}),
    'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [
        {'index': 0, 'id': 'c1',
         'function': {'name': 'read_file', 'arguments': '{"pa'}}]}}]}),
    'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [
        {'index': 0, 'function': {'arguments': 'th":"a.py"}'}}]},
        'finish_reason': 'tool_calls'}],
        'usage': {'prompt_tokens': 11, 'completion_tokens': 7,
                  'prompt_cache_hit_tokens': 5, 'prompt_cache_miss_tokens': 6}}),
    'data: [DONE]',
])

BODY = {
    'model': 'gpt-4o',
    'messages': [
        {'role': 'system', 'content': 'You are a careful coding agent.'},
        {'role': 'user', 'content': 'fix the bug in foo.py'},
    ],
    'tools': [{'type': 'function',
               'function': {'name': 'read_file', 'description': 'read a file'}}],
    'temperature': 0.2,
}


def _body(**extra: object) -> dict:
    data = json.loads(json.dumps(BODY))
    data.update(extra)
    return data


class PureTest(unittest.TestCase):
    """不碰数据库的部分：脱敏、重建、差异、指纹。"""

    def test_scrub_removes_tokens_but_stays_json(self) -> None:
        raw = json.dumps({'api_key': 'sk-abcdef0123456789', 'note': 'Bearer abcdefghijkl'})
        out = tap.scrub(raw)
        self.assertNotIn('sk-abcdef0123456789', out)
        self.assertNotIn('abcdefghijkl', out)
        # 仍然可解析：脱敏替换的是值，不是结构
        json.loads(out)

    def test_redact_headers_masks_secrets(self) -> None:
        out = tap.redact_headers([
            ('authorization', 'Bearer sk-livesecret1234'),
            ('x-api-key', 'abcd1234efgh'),
            ('cookie', 'wb_session=abcdef123456'),
            ('content-type', 'application/json'),
        ])
        for name in ('authorization', 'x-api-key', 'cookie'):
            self.assertNotIn('secret', out[name].lower())
            self.assertIn('***', out[name])
        self.assertEqual(out['content-type'], 'application/json', '普通头不该被打码')

    def test_reconstruct_merges_streaming_tool_arguments(self) -> None:
        out = tap.reconstruct(text=SSE)
        self.assertEqual(out['text'], 'Hello')
        self.assertEqual(out['reasoning'], '想一想')
        self.assertEqual(out['finish'], 'tool_calls')
        self.assertTrue(out['done'], '[DONE] 必须被识别为流结束')
        self.assertEqual(len(out['tool_calls']), 1)
        call = out['tool_calls'][0]
        self.assertEqual(call['name'], 'read_file')
        # 参数跨两帧：拼不对这里会是 '{"pa' 或非法 JSON
        self.assertEqual(json.loads(call['arguments']), {'path': 'a.py'})

    def test_reconstruct_reads_non_stream_payload(self) -> None:
        out = tap.reconstruct(payload={'choices': [{
            'message': {'content': 'done', 'tool_calls': [
                {'id': 'x', 'function': {'name': 'write_file', 'arguments': '{}'}}]},
            'finish_reason': 'stop'}]})
        self.assertEqual(out['text'], 'done')
        self.assertEqual(out['tool_calls'][0]['name'], 'write_file')

    def test_fingerprint_groups_same_run_and_separates_runs(self) -> None:
        a = _body()
        b = _body(messages=a['messages'] + [{'role': 'user', 'content': 'step 2'}])
        c = _body(messages=[a['messages'][0], {'role': 'user', 'content': '别的问题'}])
        self.assertEqual(tap._fingerprint(a, 'chat'), tap._fingerprint(b, 'chat'),
                         '同一任务追加历史后指纹必须不变，否则会话会被拆成几十条')
        self.assertNotEqual(tap._fingerprint(a, 'chat'), tap._fingerprint(c, 'chat'))

    def test_diff_reports_added_message_and_changed_param(self) -> None:
        before = {'id': 1, 'seq': 1, 'ts': 0, 'model': 'gpt-4o',
                  'request': {'body': _body()}}
        after = {'id': 2, 'seq': 2, 'ts': 0, 'model': 'gpt-4o',
                 'request': {'body': _body(
                     temperature=0.7,
                     messages=BODY['messages'] + [{'role': 'assistant', 'content': 'ok'}]),
                 }}
        # 直接调 diff 需要 turn id，这里复用内部实现：改成等价的两个假 turn
        import unittest.mock as mock
        with mock.patch.object(tap, 'turn', side_effect=[before, after]):
            out = tap.diff(1, 2)
        assert out is not None
        self.assertEqual(out['summary']['added'], 1)
        self.assertEqual([p['key'] for p in out['params']], ['temperature'])

    def test_client_family(self) -> None:
        self.assertEqual(tap._client_family('claude-code/1.0'), 'claude')
        self.assertEqual(tap._client_family('codex_cli_rs/0.9'), 'codex')
        self.assertEqual(tap._client_family('curl/8.4.0'), 'cli')
        self.assertEqual(tap._client_family(None), 'unknown')


class DbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        d = Path(cls._tmp.name)
        cls._orig = (config.DB_PATH, config.USERS_FILE, config.STATIC_DIR)
        config.DB_PATH = d / 'tap.db'
        config.USERS_FILE = d / 'users.json'
        config.STATIC_DIR = d / 'no-static'
        config.USERS_FILE.write_text(json.dumps({
            'secret': 'T' * 64,
            'users': [
                {'username': 'admin', 'role': 'admin',
                 'pwd_hash': security.make_hash('admin-pw')},
                {'username': 'viewer', 'role': 'viewer',
                 'pwd_hash': security.make_hash('viewer-pw')},
            ],
            'api_keys': [],
        }), encoding='utf-8')
        db._conn = None
        db.connect()
        from server.main import app
        from fastapi.testclient import TestClient
        cls.c = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        if db._conn is not None:
            db._conn.close()
        db._conn = None
        config.DB_PATH, config.USERS_FILE, config.STATIC_DIR = cls._orig
        try:
            cls._tmp.cleanup()
        except PermissionError:
            pass

    def setUp(self) -> None:
        security._fail.clear()
        security._user_fail.clear()
        self.c.cookies.clear()
        tap.clear()
        tap.save_settings({'enabled': False})
        db.set_setting(tap.SETTING_KEY, {'enabled': False})

    def tearDown(self) -> None:
        self.c.cookies.clear()

    @contextmanager
    def _as(self, username: str, password: str):
        r = self.c.post('/api/login', json={'username': username, 'password': password})
        self.assertEqual(r.status_code, 200, r.text)
        try:
            yield self.c
        finally:
            self.c.cookies.clear()

    def _capture(self, **kwargs) -> int | None:
        base = dict(
            body=_body(), headers=[('authorization', 'Bearer sk-live1234567890')],
            key={'id': 1}, ua='claude-code/1.0', path='/v1/chat/completions',
            protocol='chat', status=200, latency_ms=100, stream=True,
            text=SSE, usage={'prompt_tokens': 11, 'completion_tokens': 7},
        )
        base.update(kwargs)
        return tap.capture(**base)

    # ── 服务层 ───────────────────────────────────────────
    def test_disabled_writes_nothing(self) -> None:
        self.assertIsNone(self._capture())
        self.assertEqual(tap.stats()['turns'], 0)

    def test_capture_groups_turns_and_increments_seq(self) -> None:
        tap.save_settings({'enabled': True})
        self.assertIsNotNone(self._capture())
        self._capture(body=_body(messages=BODY['messages'] + [
            {'role': 'user', 'content': 'step 2'}]))
        session = tap.list_sessions()['items'][0]
        self.assertEqual(session['turns'], 2)
        detail = tap.session(session['id'])
        assert detail is not None
        self.assertEqual([t['seq'] for t in detail['items']], [1, 2])
        # 另一句话 → 另一次任务
        self._capture(body=_body(messages=[BODY['messages'][0],
                                          {'role': 'user', 'content': '别的任务'}]))
        self.assertEqual(tap.list_sessions()['total'], 2)

    def test_capture_marks_truncation(self) -> None:
        tap.save_settings({'enabled': True, 'max_body': 4096})
        big = _body(messages=BODY['messages'] + [
            {'role': 'user', 'content': 'x' * 20000}])
        turn_id = self._capture(body=big)
        assert turn_id is not None
        view = tap.turn_view(turn_id)
        assert view is not None
        self.assertTrue(view['truncated'])
        self.assertFalse(view['request_ok'], '被截断的 JSON 不该被当成完整结构')

    def test_capture_never_raises(self) -> None:
        """形状再怪的输入、再离谱的内部故障，都不能把异常抛回请求路径。"""
        tap.save_settings({'enabled': True})
        # 1) 奇怪但合法的输入：对象、空头、缺 usage —— 不抛即可（存成什么样不重要）
        tap.capture(body=object(), headers=None, key=None, ua=None, path='',
                    protocol='chat', status=0, latency_ms=None, stream=False,
                    payload=None, usage=None)
        # 2) 库写入失败：必须被吞掉并返回 None（请求已经发给客户端了）
        import unittest.mock as mock
        with mock.patch.object(db, 'execute', side_effect=RuntimeError('disk full')):
            self.assertIsNone(self._capture())

    def test_secret_never_reaches_storage(self) -> None:
        tap.save_settings({'enabled': True})
        turn_id = self._capture(headers=[('authorization', 'Bearer sk-live1234567890')])
        row = db.query_one('SELECT request_json FROM tap_turns WHERE id = ?', (turn_id,))
        assert row is not None
        self.assertNotIn('sk-live1234567890', row['request_json'])

    def test_prune_respects_max_sessions(self) -> None:
        tap.save_settings({'enabled': True, 'max_sessions': 10})
        for i in range(14):
            self._capture(body=_body(messages=[BODY['messages'][0],
                                             {'role': 'user', 'content': f'task {i}'}]))
        tap.prune()
        self.assertLessEqual(tap.list_sessions()['total'], 10)

    # ── 接口 ─────────────────────────────────────────────
    def test_api_requires_login(self) -> None:
        r = self.c.get('/api/tap/sessions')
        self.assertIn(r.status_code, (401, 403))

    def test_api_config_toggle_and_flow(self) -> None:
        with self._as('admin', 'admin-pw') as c:
            cfg = c.get('/api/tap/config')
            self.assertEqual(cfg.status_code, 200)
            self.assertFalse(cfg.json()['config']['enabled'])
            r = c.put('/api/tap/config', json={'enabled': True, 'retention_days': 3})
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.json()['config']['enabled'])
            self.assertEqual(r.json()['config']['retention_days'], 3)

        # 开关打开后，网关路径上的 capture 才会写
        self._capture()
        with self._as('admin', 'admin-pw') as c:
            listed = c.get('/api/tap/sessions').json()
            self.assertEqual(listed['total'], 1)
            sid = listed['items'][0]['id']
            detail = c.get(f'/api/tap/sessions/{sid}').json()
            self.assertEqual(len(detail['items']), 1)
            turn_id = detail['items'][0]['id']
            view = c.get(f'/api/tap/turns/{turn_id}').json()
            self.assertEqual(view['recon']['text'], 'Hello')
            diff = c.get(f'/api/tap/diff?before={turn_id}&after={turn_id}').json()
            self.assertEqual(diff['summary']['added'], 0)
            export = c.get(f'/api/tap/sessions/{sid}/export')
            self.assertEqual(export.status_code, 200)
            self.assertIn('attachment', export.headers['content-disposition'])
            self.assertIn('Hello', export.text)
            self.assertEqual(c.delete(f'/api/tap/sessions/{sid}').status_code, 200)
            self.assertEqual(c.get('/api/tap/sessions').json()['total'], 0)

    def test_viewer_can_read_switch_but_not_conversations(self) -> None:
        tap.save_settings({'enabled': True})
        self._capture()
        with self._as('viewer', 'viewer-pw') as c:
            self.assertEqual(c.get('/api/tap/config').status_code, 200)
            self.assertEqual(c.get('/api/tap/sessions').status_code, 403)
            self.assertEqual(c.put('/api/tap/config', json={'enabled': False}).status_code, 403)

    def test_missing_turn_is_404(self) -> None:
        with self._as('admin', 'admin-pw') as c:
            self.assertEqual(c.get('/api/tap/turns/999999').status_code, 404)
            self.assertEqual(c.get('/api/tap/sessions/999999').status_code, 404)


if __name__ == '__main__':
    unittest.main()
