# renders fraudvae/web/story.html to an A4 pdf: python3 scripts/story_pdf.py $PWD/fraudvae/web/story.html docs/FinFraud-development-process.pdf
import sys, asyncio
from playwright.async_api import async_playwright
src, out = sys.argv[1], sys.argv[2]
PRINT_CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { font-size: 10.5pt; }
body { background: #fff !important; color: #111 !important; max-width: none !important; }
nav, .nav, .toc-sticky, .topbar, header .links, .site-links { display: none !important; }
a { color: #1d4ed8; text-decoration: none; }
h1, h2 { break-after: avoid; }
table, pre, figure, blockquote { break-inside: avoid; }
tr { break-inside: avoid; }
pre, code { font-size: 8.5pt; white-space: pre-wrap; }
"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto("file://" + src)
        await pg.emulate_media(media="print", color_scheme="light")
        await pg.add_style_tag(content=PRINT_CSS)
        await pg.pdf(path=out, format="A4", print_background=True, display_header_footer=True,
                     header_template="<div></div>",
                     footer_template="<div style='font:8pt sans-serif;color:#666;width:100%;text-align:center'>finfraud.dev &nbsp;·&nbsp; <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                     margin={"top":"18mm","bottom":"18mm","left":"16mm","right":"16mm"})
        await b.close()
asyncio.run(main())
