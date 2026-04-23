from tiger_factors.multifactor_evaluation.reporting.html_report import render_summary_report_html


def render_summary_html(*args, **kwargs):
    return render_summary_report_html(*args, **kwargs)


__all__ = [
    "render_summary_html",
    "render_summary_report_html",
]
