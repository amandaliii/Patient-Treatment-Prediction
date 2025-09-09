import os

MIMIC_DATA_DIR = "/Users/amandali/Downloads/Mimic III"
NROWS = 500000
banned_items_file = os.path.join(MIMIC_DATA_DIR, 'banned_items.txt')

CATEGORIES = {
    'chart_events': 'chart_events',
    'input_events': 'input_events',
    'lab_events': 'lab_events',
    'microbiology_events': 'microbiology_events',
    'prescriptions': 'prescriptions',
    'procedure_events': 'procedure_events'
}

# map model categories to D_ITEMS.csv categories for label lookup
CATEGORY_TO_D_ITEMS = {
    'chartevents': 'CHART',
    'inputevents': 'INPUT',
    'labevents': 'LAB',
    'microbiologyevents': 'MICROBIOLOGY',
    'prescriptions': 'PRESCRIPTIONS',
    'procedureevents': 'PROCEDURE'
}
