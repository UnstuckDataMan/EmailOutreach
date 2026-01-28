# Outreach Email Merge Tool

A lightweight browser-based tool for generating personalised outreach emails from an Excel file.

The app allows you to:
- Upload an Excel spreadsheet
- Paste one or more email templates (“copy”)
- Automatically merge Excel data into the templates using placeholders
- Rotate multiple templates (A → B → C → A …)
- Download a clean CSV ready for upload into email tools

No local setup required for users — everything runs in the browser.

---

## How it works

### 1. Upload Excel
Upload an `.xlsx` file containing headers in the first row and data beneath.

Example headers:
- `First Name`
- `Niche`
- `Email`

### 2. Write email templates
Use placeholders wrapped in double curly brackets: Hi {{first_name}}, we see you are in the {{Niche}} space and wanted to reach out.


**Placeholder rules**
- Case-insensitive  
- Spaces and underscores are ignored  

So all of these match the same column:
- `{{first_name}}`
- `{{First Name}}`
- `{{FIRSTNAME}}`

---

## Multiple templates (rotation)

You can add as many templates as you like.

If you enter:
- Template A
- Template B
- Template C

Rows will be generated as:
- Row 1 → Template A  
- Row 2 → Template B  
- Row 3 → Template C  
- Row 4 → Template A  
- …and so on

---

## Output

The downloaded CSV will contain **only**:

- `Email` (if present in the input Excel)
- `Email Copy`

Example:

```csv
Email,Email Copy
john.doe@example.com,"Hi John, we see you are in the IT space and wanted to reach out."
jane.smith@example.com,"Hi Jane, we see you are in the Marketing space and wanted to reach out."


