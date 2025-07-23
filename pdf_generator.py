import os
import json
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime
import matplotlib.pyplot as plt
import uuid
import glob

def find_latest_reasoning_json(serial_number):
    serial_dir = os.path.join(REASON_PATH, serial_number)
    if not os.path.isdir(serial_dir):
        return ""

    # Find all timestamp subdirectories under this serial
    subdirs = [
        d for d in os.listdir(serial_dir)
        if os.path.isdir(os.path.join(serial_dir, d))
    ]

    # Parse and sort timestamps
    try:
        subdirs_sorted = sorted(
            subdirs,
            key=lambda x: datetime.strptime(x, "%Y-%m-%dT%H-%M-%S"),
            reverse=True
        )
    except ValueError:
        return ""

    for timestamp_dir in subdirs_sorted:
        reasoning_path = os.path.join(serial_dir, timestamp_dir, "reasoning.json")
        if os.path.exists(reasoning_path):
            try:
                with open(reasoning_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("reasoning", "")
            except Exception:
                continue

    return ""

def generate_pie_chart(labels, sizes, title, output_path):
    plt.figure(figsize=(4, 4))
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.total_crack_count = 0
        self.total_crack_scans = 0
        self.total_thermal_count = 0
        self.total_thermal_points = 0

    def header(self):
        if self.page_no() == 1:
            self.set_font("Helvetica", 'B', 16)
            self.cell(0, 10, "Solar Panel Inspection Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(5)

    def cover_page(self, total_panels, inspection_line, inspector, avg_crack_rate, avg_thermal_rate,
                   crack_pie_data=None, thermal_pie_data=None, suggestion=""):
        import uuid
        self.set_font("Helvetica", '', 12)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Inspection Line: {inspection_line}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Inspector: {inspector}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 10, f"Total Panels: {total_panels}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        
        self.set_font("Helvetica", 'B', 12)
        self.cell(0, 10, "Overall Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", '', 11)
        self.cell(0, 8, f"Average Crack Rate: {avg_crack_rate:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 8, f"Average Overheat Rate: {avg_thermal_rate:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


        self.ln(5)

        # Pie Charts
        chart_id = str(uuid.uuid4())[:8]
        if crack_pie_data:
            crack_chart = f"summary_crack_chart_{chart_id}.png"
            generate_pie_chart(["Cracked", "Normal"], crack_pie_data, "Overall Crack Distribution", crack_chart)
            if os.path.exists(crack_chart):
                self.image(crack_chart, x=(self.w - 80) / 2, w=80)
                os.remove(crack_chart)

        if thermal_pie_data:
            thermal_chart = f"summary_thermal_chart_{chart_id}.png"
            generate_pie_chart(["Overheat", "Normal"], thermal_pie_data, "Overall Thermal Distribution", thermal_chart)
            if os.path.exists(thermal_chart):
                self.image(thermal_chart, x=(self.w - 80) / 2, w=80)
                os.remove(thermal_chart)

        self.ln(10)

        if suggestion:
            self.ln(5)
            self.set_font("Helvetica", 'B', 11)
            self.cell(0, 8, "Suggested Solution:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", '', 11)
            self.multi_cell(0, 8, suggestion)
            
        self.ln(5)



    def product_section(self, panel):
        chart_id = str(uuid.uuid4())[:8]

        self.set_font("Helvetica", 'B', 14)
        self.cell(0, 10, f"Panel Serial: {panel.get('serial_number', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

        self.set_font("Helvetica", '', 12)
        model_name = panel.get("model_name", "N/A")
        if model_name == "unknown yet":
            model_name = "10W Solar Panel"
        self.cell(50, 8, "Model Name:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 8, model_name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.cell(50, 8, "Timestamp:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.cell(0, 8, panel.get("timestamp", "N/A"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        status = panel.get("status", "")
        if status:
            self.cell(50, 8, "Status:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(0, 8, status, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(4)

        # --- Vision Scan ---
        status_map = panel.get("status_by_timestamp", {})
        cracked_count = 0
        total = 0
        if status_map:
            timestamps = sorted(status_map.keys())
            statuses = list(status_map.values())
            cracked_count = sum(1 for s in statuses if isinstance(s, str) and s.lower() == "cracked")
            total = len(statuses)
            crack_rate = (cracked_count / total) * 100 if total > 0 else 0

            self.set_font("Helvetica", 'B', 12)
            self.cell(0, 10, "Vision Scan Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", '', 11)
            self.multi_cell(0, 8,
                f"Scan Duration: {timestamps[0]} to {timestamps[-1]}\n"
                f"Total Scans: {total}\n"
                f"Cracked Count: {cracked_count}\n"
                f"Crack Rate: {crack_rate:.2f}%"
            )

            # Pie Chart
            crack_chart_path = f"crack_chart_{chart_id}.png"
            generate_pie_chart(
                labels=["Cracked", "Normal"],
                sizes=[cracked_count, total - cracked_count],
                title="Crack Distribution",
                output_path=crack_chart_path
            )
            if os.path.exists(crack_chart_path):
                chart_width = 80
                x_centered = (self.w - chart_width) / 2
                self.image(crack_chart_path, x=x_centered, w=chart_width)
                os.remove(crack_chart_path)

            self.ln(4)

            self.total_crack_count += cracked_count
            self.total_crack_scans += total

        # --- Thermal Scan ---
        thermal_map = panel.get("thermal_by_timestamp", {})
        overheat_count = 0
        total_points = 0
        if thermal_map:
            all_timestamps = sorted(thermal_map.keys())
            for ts, areas in thermal_map.items():
                for area_grid in areas.values():
                    for row in area_grid:
                        for val in row:
                            total_points += 1
                            if val > 38:
                                overheat_count += 1

            overheat_rate = (overheat_count / total_points) * 100 if total_points > 0 else 0

            self.set_font("Helvetica", 'B', 12)
            self.cell(0, 10, "Thermal Sensor Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_font("Helvetica", '', 11)
            self.multi_cell(0, 8,
                f"Scan Duration: {all_timestamps[0]} to {all_timestamps[-1]}\n"
                f"Total Data Points: {total_points}\n"
                f"Overheated Points (>38C): {overheat_count}\n"
                f"Overheat Rate: {overheat_rate:.2f}%"
            )

            thermal_chart_path = f"thermal_chart_{chart_id}.png"
            generate_pie_chart(
                labels=["Overheat", "Normal"],
                sizes=[overheat_count, total_points - overheat_count],
                title="Thermal Distribution",
                output_path=thermal_chart_path
            )
            if os.path.exists(thermal_chart_path):
                chart_width = 80
                x_centered = (self.w - chart_width) / 2
                self.image(thermal_chart_path, x=x_centered, w=chart_width)
                os.remove(thermal_chart_path)

            self.ln(4)

            self.total_thermal_count += overheat_count
            self.total_thermal_points += total_points

        # --- Panel Image ---
        img = panel.get("image_path")
        if img and os.path.exists(img):
            try:
                self.set_font("Helvetica", 'B', 12)
                self.cell(0, 10, "Panel Image:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                self.image(img, w=100)
            except:
                self.set_font("Helvetica", 'I', 10)
                self.cell(0, 10, f"[Could not load image: {img}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            self.set_font("Helvetica", 'I', 10)
            self.cell(0, 10, "[No image available]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.add_page()
        



DATA_PATH = "Database/products.json"
REASON_PATH = "static/product_images"

# Step 1: Load product list
with open(DATA_PATH, "r", encoding="utf-8") as f:
    products = json.load(f)

# Step 2: Pre-process crack and thermal statistics
total_crack_count = 0
total_crack_scans = 0
total_thermal_count = 0
total_thermal_points = 0

for product in products:
    # Crack stats
    status_map = product.get("status_by_timestamp", {})
    cracked = sum(1 for s in status_map.values() if isinstance(s, str) and s.lower() == "cracked")
    total = len(status_map)
    total_crack_count += cracked
    total_crack_scans += total

    # Thermal stats
    thermal_map = product.get("thermal_by_timestamp", {})
    for areas in thermal_map.values():
        for area_grid in areas.values():
            for row in area_grid:
                for val in row:
                    total_thermal_points += 1
                    if val > 38:
                        total_thermal_count += 1

avg_crack_rate = (total_crack_count / total_crack_scans * 100) if total_crack_scans else 0
avg_thermal_rate = (total_thermal_count / total_thermal_points * 100) if total_thermal_points else 0


# Step 3: Create PDF and Cover Page
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()


latest_reasoning = ""
for product in products:
    serial = product.get("serial_number")
    if not serial:
        continue
    latest_reasoning = find_latest_reasoning_json(serial)
    if latest_reasoning:
        break  # stop at first found

# Add cover first
pdf.cover_page(
    total_panels=len(products),
    inspection_line="Line 1",
    inspector="Smart Conveyor Automated System",
    avg_crack_rate=avg_crack_rate,
    avg_thermal_rate=avg_thermal_rate,
    crack_pie_data=(total_crack_count, total_crack_scans - total_crack_count),
    thermal_pie_data=(total_thermal_count, total_thermal_points - total_thermal_count),
    suggestion=latest_reasoning
)

# Step 4: Add product detail pages
for product in products:
    pdf.product_section(product)

# Step 5: Output PDF
pdf.output("solar_inspection_report.pdf")

