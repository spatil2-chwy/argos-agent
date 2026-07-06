#!/usr/bin/env python3
"""Small dependency-free Markdown-to-DOCX converter for Argos VBR write-ups."""

from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def style_id(name: str) -> str:
    return name.replace(" ", "")


def run(text: str, *, bold: bool = False, code: bool = False) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if code:
        props.append('<w:rFonts w:ascii="Cascadia Mono" w:hAnsi="Cascadia Mono"/><w:sz w:val="18"/><w:szCs w:val="18"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    space = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:r>{rpr}<w:t{space}>{esc(text)}</w:t></w:r>"


INLINE_RE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*)")


def inline_runs(text: str) -> str:
    pieces = []
    pos = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > pos:
            pieces.append(run(text[pos : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            pieces.append(run(token[1:-1], code=True))
        else:
            pieces.append(run(token[2:-2], bold=True))
        pos = match.end()
    if pos < len(text):
        pieces.append(run(text[pos:]))
    return "".join(pieces)


def paragraph(text: str, style: str | None = None, indent: bool = False) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style_id(style)}"/>')
    if indent:
        ppr.append('<w:ind w:left="360" w:hanging="180"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{inline_runs(text)}</w:p>"


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = max(len(row) for row in rows)
    width = int(9000 / max(cols, 1))
    parts = [
        "<w:tbl><w:tblPr>",
        '<w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>',
        "</w:tblPr><w:tblGrid>",
    ]
    parts.extend(f'<w:gridCol w:w="{width}"/>' for _ in range(cols))
    parts.append("</w:tblGrid>")
    for row_index, row in enumerate(rows):
        parts.append("<w:tr>")
        for col_index in range(cols):
            fill = "dbeafe" if row_index == 0 else "ffffff"
            text = row[col_index] if col_index < len(row) else ""
            parts.append(
                "<w:tc><w:tcPr>"
                f'<w:tcW w:w="{width}" w:type="dxa"/><w:shd w:fill="{fill}"/>'
                "</w:tcPr>"
                + paragraph(text, "Table Text")
                + "</w:tc>"
            )
        parts.append("</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


def image_block(rel_id: str, alt: str, width_in: float, height_in: float) -> str:
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    safe_alt = html.escape(alt, quote=True)
    return f'''<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="{WP_NS}">
  <wp:extent cx="{cx}" cy="{cy}"/>
  <wp:effectExtent l="0" t="0" r="0" b="0"/>
  <wp:docPr id="1" name="{safe_alt}" descr="{safe_alt}"/>
  <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1" xmlns:a="{A_NS}"/></wp:cNvGraphicFramePr>
  <a:graphic xmlns:a="{A_NS}"><a:graphicData uri="{PIC_NS}">
    <pic:pic xmlns:pic="{PIC_NS}">
      <pic:nvPicPr><pic:cNvPr id="0" name="{safe_alt}"/><pic:cNvPicPr/></pic:nvPicPr>
      <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
      <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
    </pic:pic>
  </a:graphicData></a:graphic>
</wp:inline>
</w:drawing></w:r></w:p>'''


def parse(markdown: str) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    para: list[str] = []
    lines = markdown.splitlines()
    i = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            blocks.append(("p", " ".join(line.strip() for line in para)))
            para = []

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_para()
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush_para()
            blocks.append((f"h{len(heading.group(1))}", heading.group(2)))
            i += 1
            continue
        if line.startswith("|"):
            flush_para()
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        image = re.match(r'^!\[(.*?)\]\((.*?)\)(?:\{width=([0-9.]+)\s+height=([0-9.]+)\})?$', line)
        if image:
            flush_para()
            blocks.append(
                (
                    "image",
                    {
                        "alt": image.group(1),
                        "path": image.group(2),
                        "width": float(image.group(3) or 6.6),
                        "height": float(image.group(4) or 2.4),
                    },
                )
            )
            i += 1
            continue
        bullet = re.match(r"^-\s+(.*)$", line)
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if bullet or numbered:
            flush_para()
            blocks.append(("li", (bullet or numbered).group(1)))
            i += 1
            continue
        para.append(raw)
        i += 1
    flush_para()
    return blocks


def document_xml(blocks: list[tuple[str, object]], image_rels: dict[str, str]) -> str:
    body = []
    first_h1 = True
    for kind, value in blocks:
        if kind == "h1" and first_h1:
            body.append(paragraph(str(value), "Title"))
            first_h1 = False
        elif kind.startswith("h"):
            body.append(paragraph(str(value), f"Heading {kind[-1]}"))
        elif kind == "li":
            body.append(paragraph("- " + str(value), "List Paragraph", indent=True))
        elif kind == "table":
            body.append(table(value))  # type: ignore[arg-type]
        elif kind == "image":
            image = value  # type: ignore[assignment]
            body.append(
                image_block(
                    image_rels[str(image["path"])],
                    str(image["alt"]),
                    float(image["width"]),
                    float(image["height"]),
                )
            )
        else:
            text = str(value)
            style = "Caption" if text.startswith(("Figure ", "Table ")) else None
            body.append(paragraph(text, style))
    sect = (
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}"><w:body>'
        + "".join(body)
        + sect
        + "</w:body></w:document>"
    )


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="110" w:line="270" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="21"/><w:szCs w:val="21"/><w:color w:val="1f2328"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:sz w:val="40"/><w:szCs w:val="40"/><w:color w:val="0f766e"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="260" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/><w:color w:val="293241"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="220" w:after="70"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/><w:color w:val="2b5f96"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="Heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/></w:pPr><w:rPr><w:b/><w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="4d7c45"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="120" w:after="50"/></w:pPr><w:rPr><w:b/><w:i/><w:sz w:val="19"/><w:szCs w:val="19"/><w:color w:val="666a70"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="40" w:line="230" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="80"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/><w:left w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/><w:right w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="cbd5e1"/></w:tblBorders><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>'''


def write_docx(markdown: Path, output: Path) -> None:
    blocks = parse(markdown.read_text(encoding="utf-8"))
    image_paths = []
    for kind, value in blocks:
        if kind == "image":
            image_paths.append(str(value["path"]))  # type: ignore[index]
    image_rels = {path: f"rId{index + 2}" for index, path in enumerate(image_paths)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="svg" ContentType="image/svg+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>''')
        zf.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>''')
        rel_parts = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>''']
        for path, rel_id in image_rels.items():
            source = (markdown.parent / path).resolve()
            media_name = Path(path).name
            rel_parts.append(f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{media_name}"/>')
            zf.write(source, f"word/media/{media_name}")
        rel_parts.append("</Relationships>")
        zf.writestr("word/_rels/document.xml.rels", "".join(rel_parts))
        zf.writestr("word/document.xml", document_xml(blocks, image_rels))
        zf.writestr("word/styles.xml", styles_xml())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_docx(args.markdown, args.output)


if __name__ == "__main__":
    main()
