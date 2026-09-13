with open("Report.html", "r", encoding="utf-8") as f:
    html = f.read()

custom_css = """
<style>
  pre, code, .CodeMirror, .highlight, .output_text pre, .jp-CodeMirrorEditor {
    font-size: 11px !important;
    line-height: 1.2 !important;
  }
  .jp-RenderedImage img, img {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
  }
</style>
"""

html = html.replace("</head>", custom_css + "</head>")

with open("Report_final.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Done — open Report_final.html")