from xqgeneral.evaluate_qa import balanced_rows, normalized, qa_summary


def test_task_balance_preserves_rare_groups_and_does_not_repair_answers():
    rows = [{'id': str(i), 'stage': 'current', 'task_type': 'common'} for i in range(20)]
    rows += [{'id': 'rare', 'stage': 'future', 'task_type': 'rare'}]
    selected = balanced_rows(rows, 3, 7)
    assert len(selected) == 4 and selected == balanced_rows(rows, 3, 7)
    assert any(r['id'] == 'rare' for r in selected)
    assert normalized('  a0a1\n b0c2 ') == 'a0a1 b0c2'
    assert normalized('建议走a0a1') != normalized('a0a1')
    scores = qa_summary([dict(r, correct=(r['id'] == 'rare')) for r in selected])
    assert scores['accuracy'] == 0.25
    assert scores['by_task']['future/rare']['accuracy'] == 1.0
