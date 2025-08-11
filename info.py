MIMIC_DATA_DIR = "/Users/amandali/Downloads/Mimic III"
NROWS = 2000000

CATEGORIES = {
    'chart_events': 'chart_items',
    'input_events': 'input_items',
    'lab_events': 'lab_items',
    'microbiology_events': 'microbiology_items',
    'prescriptions': 'prescriptions_items',
    'procedure_events': 'procedure_items'
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
