import contextlib
import io
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spec2formula.cli import main
from spec2formula.output import exclusive_output

ROOT = Path(__file__).resolve().parents[1]


def competing_writer(path, ready, release, results):
    try:
        with exclusive_output(path) as stream:
            stream.write('first writer\n')
            ready.set()
            if not release.wait(15):
                raise RuntimeError('Parent did not release writer')
        results.put('saved')
    except Exception as exc:
        ready.set()
        results.put(type(exc).__name__)


class OutputTests(unittest.TestCase):
    def test_existing_output_or_partial_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            for occupied in [output, output.with_name('out.jsonl.partial')]:
                with self.subTest(occupied=occupied.name):
                    occupied.write_text('original', encoding='utf-8')
                    with self.assertRaises(FileExistsError):
                        with exclusive_output(output):
                            self.fail('Work started despite occupied path')
                    self.assertEqual(occupied.read_text(), 'original')
                    occupied.unlink()

    def test_target_created_during_work_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            with self.assertRaises(FileExistsError):
                with exclusive_output(output) as stream:
                    stream.write('our result\n')
                    output.write_text('other writer\n', encoding='utf-8')
            self.assertEqual(output.read_text(), 'other writer\n')
            self.assertEqual(output.with_name('out.jsonl.partial').read_text(), 'our result\n')

    def test_failed_work_retains_partial_without_success_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            with self.assertRaisesRegex(ValueError, 'failed record'):
                with exclusive_output(output) as stream:
                    stream.write('completed record\n')
                    raise ValueError('failed record')
            self.assertFalse(output.exists())
            self.assertEqual(output.with_name('out.jsonl.partial').read_text(), 'completed record\n')

    def test_unsupported_publication_does_not_replace_or_discard(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            with patch('spec2formula.output.os.link', side_effect=OSError('unsupported')):
                with self.assertRaisesRegex(OSError, 'unsupported'):
                    with exclusive_output(output) as stream:
                        stream.write('result\n')
            self.assertFalse(output.exists())
            self.assertEqual(output.with_name('out.jsonl.partial').read_text(), 'result\n')

    def test_multiprocess_competition(self):
        ctx = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            ready, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
            worker = ctx.Process(target=competing_writer, args=(str(output), ready, release, results))
            worker.start()
            try:
                self.assertTrue(ready.wait(15))
                with self.assertRaises(FileExistsError):
                    with exclusive_output(output):
                        self.fail('Second process obtained the reservation')
            finally:
                release.set()
                worker.join(20)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            self.assertEqual(worker.exitcode, 0)
            self.assertEqual(results.get(timeout=5), 'saved')
            self.assertEqual(output.read_text(), 'first writer\n')
            self.assertFalse(output.with_name('out.jsonl.partial').exists())
            results.close()

    def test_cli_reserves_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'out.jsonl'
            def fail_loading(*args, **kwargs):
                self.assertTrue(output.with_name('out.jsonl.partial').is_file())
                raise ValueError('load failure')
            with patch('spec2formula.predictor.Predictor', side_effect=fail_loading), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['predict', '--input', str(ROOT/'examples/synthetic.jsonl'), '--output', str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
