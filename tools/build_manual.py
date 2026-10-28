"""Build the Chinese user manual without importing ArrayLab or Qt.

Document-only dependencies: reportlab and pypdf. The OFL font is kept beside
the editable manual source; ReportLab embeds only the glyphs actually used.
"""
from pathlib import Path
import argparse
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether,Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
FONT=ROOT/'docs/manual/fonts/NotoSansSC-Regular.ttf'
BLUE=colors.HexColor('#195778');TEAL=colors.HexColor('#168b9c');INK=colors.HexColor('#233746')

def inline(value):
    value=html.escape(value)
    value=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#155f87">\1</link>',value)
    value=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',value)
    value=re.sub(r'`([^`]+)`',r'<font color="#255d72">\1</font>',value)
    return value

def build(source,output,version):
    if not FONT.is_file():raise FileNotFoundError('Missing OFL manual font: '+str(FONT))
    pdfmetrics.registerFont(TTFont('ManualCJK',str(FONT)))
    pdfmetrics.registerFontFamily('ManualCJK',normal='ManualCJK',bold='ManualCJK',italic='ManualCJK',boldItalic='ManualCJK')
    style={
      'body':ParagraphStyle('body',fontName='ManualCJK',fontSize=12,leading=18.8,textColor=INK,spaceAfter=8,wordWrap='CJK'),
      'title':ParagraphStyle('title',fontName='ManualCJK',fontSize=28,leading=40,textColor=BLUE,spaceAfter=18),
      'h2':ParagraphStyle('h2',fontName='ManualCJK',fontSize=20,leading=29,textColor=BLUE,spaceAfter=14,keepWithNext=True),
      'h3':ParagraphStyle('h3',fontName='ManualCJK',fontSize=12.2,leading=18,textColor=TEAL,spaceBefore=9,spaceAfter=6,keepWithNext=True),
      'cell':ParagraphStyle('cell',fontName='ManualCJK',fontSize=10.4,leading=15.3,wordWrap='CJK',textColor=INK),
      'toc':ParagraphStyle('toc',fontName='ManualCJK',fontSize=9.1,leading=13.2,wordWrap='CJK',textColor=INK),
      'head':ParagraphStyle('head',fontName='ManualCJK',fontSize=10.4,leading=15.3,wordWrap='CJK',textColor=colors.white),
      'code':ParagraphStyle('code',fontName='ManualCJK',fontSize=9.1,leading=13.5,wordWrap='CJK',backColor=colors.HexColor('#edf3f7'),borderPadding=7,spaceAfter=10),
      'caption':ParagraphStyle('caption',fontName='ManualCJK',fontSize=8.5,leading=12.5,textColor=BLUE,spaceBefore=5,spaceAfter=7,alignment=TA_CENTER,wordWrap='CJK'),
    }
    text=source.read_text(encoding='utf-8').replace('{{VERSION}}',version)
    sections=text.split('<!-- page -->');total=len(sections)
    headings=[next((line[3:] for line in section.splitlines() if line.startswith('## ')), '') for section in sections]
    def para(value,kind='body'):return Paragraph(inline(value),style[kind])
    story=[]
    for page,section in enumerate(sections,1):
        if page>1:story.append(PageBreak())
        lines=section.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            picture=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',line)
            if picture:
                path=(source.parent/picture.group(2)).resolve()
                if not path.is_file():raise FileNotFoundError(path)
                item=Image(str(path));item._restrictSize(A4[0]-92,290)
                story.append(KeepTogether([Spacer(1,6),item,para(picture.group(1),'caption')]))
                i+=1;continue
            if line=='{{TOC}}':
                rows=[]
                for n,title in enumerate(headings,1):
                    if n>2:rows.append([para(title,'toc'),para(str(n),'toc')])
                table=Table(rows,colWidths=[A4[0]-128,36]);table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),2),('LINEBELOW',(0,0),(-1,-1),.25,colors.HexColor('#dce5eb'))]));story.append(table);i+=1;continue
            if line.startswith('```'):
                code=[];i+=1
                while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
                story.append(Paragraph('<br/>'.join(html.escape(x).replace(' ','&#160;') for x in code),style['code']));i+=1;continue
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                    i+=1
                columns=len(rows[0]);width=A4[0]-92
                widths=([width*.29,width*.71] if columns==2 else [width/columns]*columns)
                wrapped=[[para(c,'head' if r==0 else 'cell') for c in row] for r,row in enumerate(rows)]
                table=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f5f8')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d5e1e8'))]))
                story.extend([table,Spacer(1,8)]);continue
            if line.startswith('# '):story.extend([Spacer(1,50),para(line[2:],'title')]);i+=1;continue
            if line.startswith('## '):story.append(para(line[3:],'h2'));i+=1;continue
            if line.startswith('### '):story.append(para(line[4:],'h3'));i+=1;continue
            if line.startswith('- '):line='• '+line[2:]
            story.append(para(line));i+=1
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=54,bottomMargin=46,title=f'ArrayLab {version} 软件使用说明书',author='ArrayLab 项目',pageCompression=1)
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('ManualCJK',8);canvas.setFillColor(BLUE)
        canvas.drawString(46,A4[1]-29,'ARRAYLAB  |  软件使用说明书')
        canvas.setStrokeColor(colors.HexColor('#c7d9e4'));canvas.line(46,A4[1]-36,A4[0]-46,A4[1]-36)
        canvas.drawString(46,25,f'版本 {version} · 2026-10-28');canvas.drawRightString(A4[0]-46,25,f'{doc.page} / {total}')
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(output)
    if len(reader.pages)!=total:raise RuntimeError(f'Layout overflow: expected {total} pages, got {len(reader.pages)}; shorten or repartition the source.')
    for n,page in enumerate(reader.pages,1):
        extracted=page.extract_text() or ''
        if len(extracted.strip())<30:raise RuntimeError(f'Empty or non-searchable page: {n}')
    print(f'{output} | {total} pages | searchable Chinese text verified')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'docs/manual/使用说明书.md')
    parser.add_argument('--version',default='0.27.0')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();build(args.source,args.output or ROOT/f'output/pdf/ArrayLab-{args.version}-使用说明书.pdf',args.version)
