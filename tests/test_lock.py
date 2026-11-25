"""test 세트 정답 라벨 잠금 (eval/lock.py, 4단계 4-2)."""

from eval.lock import file_hashes, lock_errors, write_lock


def test_real_test_set_matches_lock():
    """저장소의 test 시나리오는 잠금과 같아야 한다. 바꾸려면 사용자 확정 뒤 잠금을 다시 쓴다."""
    assert lock_errors() == []


def test_lock_detects_changed_added_and_missing_files(tmp_path):
    test_dir, lock = tmp_path / "test", tmp_path / "lock.yaml"
    test_dir.mkdir()
    (test_dir / "a.yaml").write_text("id: a\n", encoding="utf-8")
    (test_dir / "b.yaml").write_text("id: b\n", encoding="utf-8")
    assert lock_errors(test_dir, lock) == ["test 세트 잠금 파일이 없다: lock.yaml"]
    written = write_lock("테스트", test_dir, lock)
    assert written["files"] == file_hashes(test_dir) and lock_errors(test_dir, lock) == []

    (test_dir / "a.yaml").write_text("id: a\nchanged: true\n", encoding="utf-8")
    (test_dir / "b.yaml").unlink()
    (test_dir / "c.yaml").write_text("id: c\n", encoding="utf-8")
    assert lock_errors(test_dir, lock) == [
        "a.yaml: 잠근 뒤에 내용이 바뀌었다", "b.yaml: 잠근 파일인데 없다", "c.yaml: 잠그지 않은 파일이다"]


def test_grader_refuses_test_results_when_lock_differs(monkeypatch):
    """test 결과는 잠근 라벨로만 채점한다. 잠금과 다르면 채점을 멈춘다."""
    import pytest

    from eval.grader import grade

    monkeypatch.setattr(grade, "lock_errors", lambda: ["a.yaml: 잠근 뒤에 내용이 바뀌었다"])
    with pytest.raises(grade.GradeError, match="잠금과 다르다"):
        grade.check_basis({"run": {"scenario": "SC-S1-03", "split": "test"}}, grade.load_seed())
    # dev 결과는 잠금과 상관없다
    grade.check_basis({"run": {"scenario": "SC-S1-01", "split": "dev"}}, grade.load_seed())


def test_unreadable_file_is_reported_and_others_still_compared(tmp_path):
    """읽지 못한 파일은 오류로 알리고 나머지 비교는 계속한다. 잠금을 쓸 때는 멈춘다 (PR #26 리뷰)."""
    import pytest

    test_dir, lock = tmp_path / "test", tmp_path / "lock.yaml"
    test_dir.mkdir()
    (test_dir / "a.yaml").write_text("id: a\n", encoding="utf-8")
    write_lock("테스트", test_dir, lock)
    (test_dir / "a.yaml").write_text("id: a\nchanged: true\n", encoding="utf-8")
    (test_dir / "dir.yaml").mkdir()  # 이름만 .yaml인 디렉터리는 읽을 수 없다
    errors = lock_errors(test_dir, lock)
    assert errors[0].startswith("dir.yaml: 파일을 읽을 수 없다")
    assert "a.yaml: 잠근 뒤에 내용이 바뀌었다" in errors
    with pytest.raises(OSError):
        write_lock("테스트", test_dir, lock)
