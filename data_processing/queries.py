


def xbrl_reports_query(date_start, date_end, result_size=100) -> dict:
    """
    Query to fetch all financial reports in XBRL format from the Virk API published in a given date range.
    Note: Will fetch from midnight of the start date to 23:59:59 of the end date.
    :param date_start: str, start date of the query in the format "YYYY-MM-DD"
    :param date_end: str, end date of the query in the format "YYYY-MM-DD"
    """

    xbrl_reports = { 
        "_source":[
            "cvrNummer", 
            "dokumenter.dokumentUrl",
            "dokumenter.dokumentMimeType",
            "offentliggoerelsesTidspunkt" 
        ], 
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "dokumenter.dokumentMimeType": "application"
                        }
                    },
                    {
                        "term": {
                            "dokumenter.dokumentMimeType": "xml"
                        }
                    },
                    {
                        "range": {
                            "offentliggoerelsesTidspunkt": {
                                "gte": f"{date_start}T00:00:00.001Z",
                                "lte": f"{date_end}T23:59:59.505Z"
                            }
                        }
                    }
                ],
                "must_not": [],
                "should": []
            }
        },
    "size": result_size
    }

    return xbrl_reports


def cvr_query(cvr_list, result_size=100) -> dict:
    """
    Query to fetch all entries matching a list of CVR numbers.
    :param cvr_list: list[str], the CVR number of the company to fetch
    """

    query = {
        "query":{"terms":{"Vrvirksomhed.cvrNummer": cvr_list}},
        "size": result_size
    }
    return query


def capital_changes_query(cvr_list, result_size=100) -> dict:
    query = {
        "query": {
            "bool": {
                "must": [{
                    "terms": {"cvrNummer": cvr_list}
                }],
                "filter": {
                    "query_string": {
                        "default_field": "virksomhedsregistreringstatusser.keyword",
                        "query": "AENDRING_KAPITAL"
                        }
                    }
                }
            },
        "size": result_size
        }

    return query