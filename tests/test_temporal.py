from chronofuse.metadata.temporal import extract_temporal, year_overlap


def test_explicit_year_and_month():
    tp = extract_temporal("The fork happened in August 2017 after the 2016 debate.")
    assert 2017 in tp.years and 2016 in tp.years
    assert tp.granularity == "month"


def test_range_and_decade():
    tp = extract_temporal("Between 2014-2016 and during the 1990s prices moved.")
    assert {2014, 2015, 2016}.issubset(tp.years)
    assert 1995 in tp.years


def test_numbers_are_not_years():
    tp = extract_temporal("It costs 2000 USD, block size is 1024 bytes and 1500.5 units.")
    assert 2000 not in tp.years and 1024 not in tp.years


def test_relative_expressions():
    tp = extract_temporal("Two weeks ago the developers asked; right now it is slow.")
    assert any("two weeks ago" in r for r in tp.relative)


def test_year_overlap_tolerance():
    assert year_overlap([2017], [2018], tolerance=1) == 1.0
    assert year_overlap([2017], [2020], tolerance=1) == 0.0
    assert year_overlap([], [2020]) == 0.0
