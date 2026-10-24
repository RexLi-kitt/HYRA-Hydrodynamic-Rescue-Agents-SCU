from pathlib import Path
from io import BytesIO
from xml.sax.saxutils import escape
from docx import Document
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
from PIL import Image as PILImage, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/simsun.ttc'))
styles={
 'body':ParagraphStyle('body',fontName='CN',fontSize=11,leading=17,spaceAfter=7,wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='CN',fontSize=23,leading=30,spaceAfter=8,keepWithNext=True),
 'subtitle':ParagraphStyle('subtitle',fontName='CN',fontSize=17,leading=23,spaceAfter=14,keepWithNext=True),
 'h1':ParagraphStyle('h1',fontName='CN',fontSize=15,leading=21,spaceBefore=12,spaceAfter=7,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='CN',fontSize=12,leading=18,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='CN',fontSize=9.5,leading=14,spaceAfter=10,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=9.5,leading=13,wordWrap='CJK'),
}
doc=Document(ROOT/'Merewether洪水对照实验结果报告.docx')
story=[]
WIDTH=526
def safe(text):
 return escape(text.replace('3×10⁻¹¹', '0.00000000003'))
for item in doc.element.body:
 if item.tag.endswith('}p'):
  p=WordParagraph(item,doc)
  blips=item.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
  if blips:
   for b in blips:
    rid=b.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    data=doc.part.related_parts[rid].blob
    im=PILImage.open(BytesIO(data));w,h=im.size
    story.append(Image(BytesIO(data),width=WIDTH,height=WIDTH*h/w))
  elif p.text.strip():
   name=p.style.name
   sty={'Title':'title','Subtitle':'subtitle','Heading 1':'h1','Heading 2':'h2','Caption':'caption'}.get(name,'body')
   story.append(Paragraph(safe(p.text),styles[sty]))
 elif item.tag.endswith('}tbl'):
  wt=WordTable(item,doc);n=len(wt.columns)
  widths=([120]+[(WIDTH-120)/(n-1)]*(n-1)) if n>2 else [110,WIDTH-110]
  rows=[[Paragraph(safe(c.text),styles['cell']) for c in row.cells] for row in wt.rows]
  t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([
   ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF1F4')),
   ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#BAC2C9')),
   ('VALIGN',(0,0),(-1,-1),'TOP'),
   ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
   ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
  ]))
  story.extend([t,Spacer(1,8)])
out=ROOT/'Merewether洪水对照实验结果报告.pdf'
def page(canvas,doc):
 canvas.setFont('CN',9);canvas.setFillColor(colors.HexColor('#606A73'))
 canvas.drawRightString(569,24,f'HYRA  |  {doc.page}')
pdf=SimpleDocTemplate(str(out),pagesize=letter,leftMargin=43,rightMargin=43,topMargin=40,bottomMargin=42,
 title='Merewether对照实验结果报告',author='HYRA项目组')
pdf.build(story,onFirstPage=page,onLaterPages=page)
qa=ROOT/'standalone/pdf-qa';qa.mkdir(parents=True,exist_ok=True)
rendered=pdfium.PdfDocument(str(out));thumbs=[]
for i in range(len(rendered)):
 page=rendered[i];im=page.render(scale=1.5).to_pil();im.save(qa/f'page-{i+1}.png')
 thumb=im.copy();thumb.thumbnail((306,396));tile=PILImage.new('RGB',(326,426),'#DADDE0');tile.paste(thumb,(10,20));ImageDraw.Draw(tile).text((10,5),f'Page {i+1}',fill='black');thumbs.append(tile)
cols=3;rows=(len(thumbs)+cols-1)//cols;sheet=PILImage.new('RGB',(cols*326,rows*426),'white')
for i,t in enumerate(thumbs):sheet.paste(t,((i%cols)*326,(i//cols)*426))
sheet.save(qa/'contact.png')
print('PDF pages',len(rendered),'size bytes',out.stat().st_size)
