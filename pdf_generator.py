import os
import json
import glob
import uuid
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ── Configuration ───────────────────────────────────────────────────────────────
DATA_PATH   = "Database/products.json"
REASON_PATH = "static/product_images"

# ── Helpers ─────────────────────────────────────────────────────────────────────

def find_latest_reasoning_json(serial_number):
    """
    Look in static/product_images/<serial_number>/*/reasoning.json,
    return the most recent 'reasoning' field if present.
    """
    base_dir = os.path.join(REASON_PATH, serial_number)
    if not os.path.isdir(base_dir):
        return ""

    # gather timestamp subdirs
    candidates = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    try:
        # ISO format: YYYY-MM-DDTHH-MM-SS
        candidates.sort(key=lambda t: datetime.strptime(t, "%Y-%m-%dT%H-%M-%S"), reverse=True)
    except ValueError:
        return ""

    for ts in candidates:
        path = os.path.join(base_dir, ts, "reasoning.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("reasoning", "")
            except Exception:
                continue
    return ""

def generate_pie_chart(labels, sizes, title, output_path):
    """
    Create a small pie chart PNG to embed in the PDF.
    """
    plt.figure(figsize=(4,4))
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ── PDFReport Class ────────────────────────────────────────────────────────────

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        # existing stats...
        self.period_start = ''
        self.period_end = ''
        self._first_panel = True

    def header(self):
        # — top‑right italic “No Changes Allowed” on every page —
        self.set_font('Helvetica', 'I', 10)
        # cell width=0 -> extend to right margin; new_x/new_y move to LMARGIN & NEXT line
        self.cell(0, 5, '"No Changes Allowed"',
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                align='R')

        # — your existing first‑page title & period —
        if self.page_no() == 1:
            # set title color to navy blue
            self.set_text_color(0, 0, 128)
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, 'Solar Panel Inspection Report',
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            # reset color back to black for everything else
            self.set_text_color(0, 0, 0)
            self.ln(5)
            if self.period_start and self.period_end:
                self.set_font('Helvetica', '', 12)
                self.cell(0, 8,
                        f'Reporting Period: {self.period_start} to {self.period_end}',
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.ln(5)

    def footer(self):
        # — bottom‑center page number —
        self.set_y(-15)                 # 15 mm from bottom
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    def set_period(self, start: str, end: str):
        self.period_start = start
        self.period_end = end


 

    def cover_page(self, total_panels, inspection_line, inspector,
                   avg_crack_rate, avg_thermal_rate,
                   crack_pie_data=None, thermal_pie_data=None):
       
        # ── Metadata ────────────────────────────────
        self.set_font("Helvetica", '', 12)
        self.cell(0, 10,
                  f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Inspection Line: {inspection_line}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Inspector: {inspector}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Total Panels: {total_panels}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

        # ── Overall Summary ─────────────────────────
        self.set_font("Helvetica", 'B', 12)
        self.cell(0, 10, "Overall Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", '', 11)
        self.cell(0, 8, f"Average Crack Rate: {avg_crack_rate:.2f}%",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 8, f"Average Overheat Rate: {avg_thermal_rate:.2f}%",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

        # ── Pie Charts ──────────────────────────────
        chart_id = str(uuid.uuid4())[:8]
        if (isinstance(crack_pie_data, (list, tuple)) and
            len(crack_pie_data) == 2 and
            all(isinstance(x, (int, float)) for x in crack_pie_data) and
            sum(crack_pie_data) > 0):
            crack_chart = f"pie_crack_{chart_id}.png"
            generate_pie_chart(["Cracked", "Normal"], crack_pie_data,
                               "Crack Distribution", crack_chart)
            if os.path.exists(crack_chart):
                self.image(crack_chart, x=(self.w-80)/2, w=80)
                os.remove(crack_chart)

        if thermal_pie_data:
            therm_chart = f"pie_thermal_{chart_id}.png"
            generate_pie_chart(["Overheat", "Normal"], thermal_pie_data,
                               "Thermal Distribution", therm_chart)
            if os.path.exists(therm_chart):
                self.image(therm_chart, x=(self.w-80)/2, w=80)
                os.remove(therm_chart)
        self.ln(10)
        # ── Product Details Summary ───────────────────
        if getattr(self, 'product_summary', None):
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 8, 'Product Details Summary',
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font('Helvetica', '', 11)
            safe = self.product_summary.encode('latin-1', errors='replace').decode('latin-1')
            self.multi_cell(0, 8, safe)
            self.ln(5)

        # ── Recommendation Box (fills to bottom margin) ───────────
        if getattr(self, 'recommendation', None):
            x = self.l_margin
            y = self.get_y()
            w = self.w - 2 * self.l_margin

            # Prepare and wrap text exactly to inner box width (w - 4 mm)
            self.set_font('Helvetica', '', 11)
            safe_reco = self.recommendation.encode('latin-1', errors='replace').decode('latin-1')
            max_text_width = w - 4
            lines = []
            for paragraph in safe_reco.split('\n'):
                words, line = paragraph.split(' '), ''
                for word in words:
                    test = f"{line} {word}".strip()
                    if self.get_string_width(test) <= max_text_width:
                        line = test
                    else:
                        lines.append(line)
                        line = word
                if line:
                    lines.append(line)
            if safe_reco.endswith('\n'):
                lines.append('')

            # Compute box height: header + wrapped lines + padding
            header_height = 6    # mm
            text_height   = 6    # mm per wrapped line
            padding_top   = 4    # mm
            padding_bot   = 4    # mm
            box_height = padding_top + header_height + len(lines) * text_height + padding_bot

            # Draw green box
            self.set_draw_color(0, 128, 0)
            self.set_fill_color(212, 237, 218)
            self.rect(x, y, w, box_height, style='DF')

            # Write header + lines inside
            self.set_xy(x + 2, y + padding_top)
            self.set_font('Helvetica', 'B', 12)
            self.cell(w - 4, header_height, 'Recommendation', ln=True, align='L')
            self.set_font('Helvetica', '', 11)
            for line in lines:
                self.cell(w - 4, text_height, line, ln=True, align='L')

            # Move cursor to just below the box
            self.set_xy(self.l_margin, y + box_height + 2)


        # ── Divider ─────────────────────────────────
        self.set_font('Courier', '', 10)
        width = self.w - 2*self.l_margin
        n_chars = int(width / self.get_string_width('='))
        divider = '=' * n_chars
        self.cell(0, 5, divider,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)





    def product_section(self, panel):
        """
        One page per panel: serial, model, timestamp, status,
        plus vision‐scan & thermal‐scan summaries and image.
        """
        self.add_page()
        if not self._first_panel:
            self.set_font('Courier', '', 10)
            width = self.w - 2*self.l_margin
            n_chars = int(width / self.get_string_width('-'))
            divider = '-' * n_chars
            self.cell(0, 5, divider,
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)
        self._first_panel = False
        chart_id = str(uuid.uuid4())[:8]

        # — Panel Header —
        self.set_font("Helvetica", 'B', 14)
        self.cell(0, 10, f"Panel Serial: {panel.get('serial_number','N/A')}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

        # Model, Timestamp, Status
        self.set_font("Helvetica", '', 12)
        self.cell(50, 8, "Model Name:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 8, panel.get("model_name", "N/A"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(50, 8, "Timestamp:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 8, panel.get("timestamp","N/A"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if panel.get("status"):
            self.cell(50, 8, "Status:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(0, 8, panel["status"],
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

        # — Vision Scan Summary —
        status_map = panel.get("status_by_timestamp", {})
        cracked_count = sum(1 for s in status_map.values()
                             if isinstance(s, str) and s.lower()=="cracked")
        total_scans = len(status_map)
        if total_scans > 0:
            timestamps = sorted(status_map.keys())
            crack_rate = cracked_count / total_scans * 100

            self.set_font("Helvetica", 'B', 12)
            self.cell(0, 10, "Vision Scan Summary",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", '', 11)
            self.multi_cell(0, 8,
                f"Scan Duration: {timestamps[0]} to {timestamps[-1]}\n"
                f"Total Scans: {total_scans}\n"
                f"Cracked Count: {cracked_count}\n"
                f"Crack Rate: {crack_rate:.2f}%"
            )

            # Pie chart
            crack_chart = f"crack_chart_{chart_id}.png"
            generate_pie_chart(
                ["Cracked", "Normal"],
                [cracked_count, total_scans - cracked_count],
                "Crack Distribution",
                crack_chart
            )
            if os.path.exists(crack_chart):
                x = (self.w - 80) / 2
                self.image(crack_chart, x=x, w=80)
                os.remove(crack_chart)
            self.ln(4)

        # — Thermal Sensor Summary —
        thermal_map = panel.get("thermal_by_timestamp", {})
        overheat = 0
        points = 0
        for areas in thermal_map.values():
            for grid in areas.values():
                for row in grid:
                    for v in row:
                        points += 1
                        if v > 38:
                            overheat += 1
        if points > 0:
            overheat_rate = overheat / points * 100

            self.set_font("Helvetica", 'B', 12)
            self.cell(0, 10, "Thermal Sensor Summary",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", '', 11)
            # Determine scan duration
            ts_list = sorted(thermal_map.keys())
            self.multi_cell(0, 8,
                f"Scan Duration: {ts_list[0]} to {ts_list[-1]}\n"
                f"Total Data Points: {points}\n"
                f"Overheated Points (>38°C): {overheat}\n"
                f"Overheat Rate: {overheat_rate:.2f}%"
            )

            # Pie chart
            therm_chart = f"thermal_chart_{chart_id}.png"
            generate_pie_chart(
                ["Overheat", "Normal"],
                [overheat, points - overheat],
                "Thermal Distribution",
                therm_chart
            )
            if os.path.exists(therm_chart):
                x = (self.w - 80) / 2
                self.image(therm_chart, x=x, w=80)
                os.remove(therm_chart)
            self.ln(4)

        # — Panel Images — 
        imgs = panel.get("image_paths", [])
        if imgs:
            self.set_font("Helvetica", 'B', 12)
            self.cell(0, 10, "Panel Images:", ln=True)
            self.ln(2)

            # calculate image size & spacing
            per_row = 3
            gutter  = 5  # space between images
            img_w   = (self.w - 2*self.l_margin - (per_row-1)*gutter) / per_row
            img_h   = img_w  # make them square; adjust if you prefer original aspect

            for idx, img_path in enumerate(imgs):
                # compute x,y for this slot
                col = idx % per_row
                if col == 0:
                    y = self.get_y()
                x = self.l_margin + col*(img_w + gutter)

                # draw image (or placeholder text on error)
                try:
                    self.image(img_path, x=x, y=y, w=img_w, h=img_h)
                except Exception:
                    self.set_xy(x, y)
                    self.set_font("Helvetica", 'I', 10)
                    self.multi_cell(img_w, 8, "[Image load error]", align='C')

                # when we’ve placed the last in a row, advance to next line
                if col == per_row - 1:
                    self.ln(img_h + gutter)
            # if the last row wasn’t full, still move down
            if len(imgs) % per_row != 0:
                self.ln(img_h + gutter)
        else:
            self.set_font("Helvetica", 'I', 10)
            self.cell(0, 10, "[No images available]", ln=True)

        self.ln(4)


        # — Inspection Reasoning —
        reasoning = find_latest_reasoning_json(panel.get('serial_number',''))
        if reasoning:
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, 'Inspection Reasoning:', ln=True)
            self.set_font('Helvetica', '', 11)
            safe = reasoning.encode('latin-1', errors='replace').decode('latin-1')
            self.multi_cell(0, 8, safe)
            self.ln(5)

    

# ── Report Generation Entry Point ──────────────────────────────────────────────

def generate_report(period: str,
                    inspection_line='Line 1',
                    inspector='Automated System',
                    period_bounds=None,
                    product_summary: str = '',
                    recommendation: str   = '') -> str:
    now = datetime.now()
    if period == 'daily': cutoff = now - timedelta(days=1)
    elif period == 'weekly': cutoff = now - timedelta(weeks=1)
    elif period == 'monthly': cutoff = now - timedelta(days=30)
    else: raise ValueError(f'Unknown period: {period}')

    start_str, end_str = ('','')
    if period_bounds:
        start_str, end_str = period_bounds
    else:
        start_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
        end_str   = now.strftime('%Y-%m-%d %H:%M:%S')

    # 1) load all products
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    # 2) filter by cutoff
    filtered = []
    for p in products:
        sbt = {}
        for ts, st in p.get("status_by_timestamp", {}).items():
            # parse folder‑style timestamp: YYYY‑MM‑DDTHH‑MM‑SS
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H-%M-%S")
            except ValueError:
                continue
            if dt >= cutoff:
                sbt[ts] = st

        tbt = {}
        for ts, areas in p.get("thermal_by_timestamp", {}).items():
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H-%M-%S")
            except ValueError:
                continue
            if dt >= cutoff:
                tbt[ts] = areas
            # **Re‑add this:**
        if sbt or tbt:
            panel = p.copy()
            panel['status_by_timestamp']  = sbt
            panel['thermal_by_timestamp'] = tbt
            panel['timestamp'] = max(list(sbt.keys()) + list(tbt.keys()))
            serial = panel["serial_number"]
            ts     = panel['timestamp']
            img_dir = os.path.join("static", "product_images", serial, ts)

# collect all image files in that folder
        panel['image_paths'] = []
        if os.path.isdir(img_dir):
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff')):
                    panel['image_paths'].append(os.path.join(img_dir, fn))

        filtered.append(panel)



    # 3) compute summary stats
    total_panels = len(filtered)
    total_scans  = sum(len(p["status_by_timestamp"]) for p in filtered)
    total_cracked = sum(
        1 for p in filtered
        for st in p["status_by_timestamp"].values()
        if (isinstance(st, str) and st.lower() == "cracked")
            or (isinstance(st, list) and any(s.lower() == "cracked" for s in st))
    )

    avg_crack_rate   = (total_cracked/total_scans*100) if total_scans else 0

    total_points= total_over = 0
    for p in filtered:
        for areas in p["thermal_by_timestamp"].values():
            for grid in areas.values():
                for row in grid:
                    total_points += len(row)
                    total_over   += sum(1 for v in row if v>38)
    avg_thermal_rate = (total_over/total_points*100) if total_points else 0

    # 4) prepare output location
    out_dir = os.path.join("Database", period)
    os.makedirs(out_dir, exist_ok=True)
    fname   = f"report_{period}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    out_path= os.path.join(out_dir, fname)

    # 5) build the PDF
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_period(start_str, end_str)
    pdf.product_summary = product_summary
    pdf.recommendation  = recommendation
    pdf.add_page()
   

    # grab one reasoning example
    pdf.cover_page(
        total_panels=total_panels,
        inspection_line=inspection_line,
        inspector=inspector,
        avg_crack_rate=avg_crack_rate,
        avg_thermal_rate=avg_thermal_rate,
        crack_pie_data=(total_cracked, total_scans-total_cracked),
        thermal_pie_data=(total_over, total_points-total_over)
    )

    for panel in filtered:
        pdf.product_section(panel)

    # 6) write file and return path
    pdf.output(out_path)
    return out_path
