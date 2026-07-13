from probe.schema import CorpusRecord


def test_corpus_record_roundtrip():
    rec = CorpusRecord(
        function_id="f", src_ir="define i32 @f() {\n  ret i32 0\n}", n_instructions=1
    )
    again = CorpusRecord.model_validate_json(rec.model_dump_json())
    assert again == rec


def test_size_buckets():
    assert CorpusRecord(function_id="a", src_ir="x", n_instructions=10).size_bucket() == "<=20"
    assert CorpusRecord(function_id="b", src_ir="x", n_instructions=30).size_bucket() == "20-50"
    assert CorpusRecord(function_id="c", src_ir="x", n_instructions=100).size_bucket() == "50-150"
    assert CorpusRecord(function_id="d", src_ir="x", n_instructions=500).size_bucket() == ">150"
