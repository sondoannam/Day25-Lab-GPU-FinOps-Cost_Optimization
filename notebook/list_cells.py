import json
import sys

def main():
    try:
        with open('/Users/sondoannam/vinuni/Day25-Lab-GPU-FinOps-Cost_Optimization/notebook/gpu_finops_lab.ipynb', 'r') as f:
            nb = json.load(f)

        for i, cell in enumerate(nb['cells']):
            source = ''.join(cell.get('source', []))
            lines = source.split('\n')
            first_line = lines[0] if lines else ''
            print(f"Cell {i} [{cell['cell_type']}]: {first_line}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
