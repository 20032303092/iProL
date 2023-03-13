import pandas as pd
import os, sys

xlsx_file = 'dataset.xlsx'
dataset = pd.DataFrame()

if not os.path.exists(xlsx_file):
    pd.DataFrame().to_excel(xlsx_file)
    print('created ', xlsx_file)

# read promoters
fasta_file = './promoter.fasta'
with open(fasta_file, 'r') as f:
    lines = f.read().splitlines()
    lines_1 = lines[0::2]
    lines_2 = lines[1::2]
    rows_pd = []
    header = ['Id', 'Pro_name', 'DNA_strand', 'TSS', 'Con_level', 'Sigma', 'Pro_seq']
    for line_1, line_2 in zip(lines_1, lines_2):
        row = line_1[1:].split(' ')
        row.append(line_2)
        rows_pd.append(row)
    promoters_pd = pd.DataFrame(rows_pd, columns=header)

# read non_promoter
fasta_file = './non_promoter.fasta'
with open(fasta_file, 'r') as f:
    lines = f.read().splitlines()
    lines_1 = lines[0::2]
    lines_2 = lines[1::2]
    rows_pd = []
    header = ['Id', 'Pro_name', 'DNA_strand', 'TSS', 'Con_level', 'Sigma', 'Pro_seq']
    for line_1, line_2 in zip(lines_1, lines_2):
        row = line_1[1:].split(' ')
        row.append("")
        row.append("")
        row.append("")
        row.append("")
        row.append("non_pro")
        row.append(line_2)
        rows_pd.append(row)
    no_promoters_pd = pd.DataFrame(rows_pd, columns=header)

dataset = promoters_pd.append(no_promoters_pd)


def write_data_dp_2_xlsx(data_pd, xlsx_file, sheet_name):
    with pd.ExcelWriter(xlsx_file, engine='openpyxl', mode='a') as writer:
        workbook = writer.book
        workbook_sheet_names = workbook.sheetnames
        if sheet_name in workbook_sheet_names:
            workbook.remove(workbook[sheet_name])
            print(sheet_name + ' has existed,removed!')
        header = ['Id', 'Pro_name', 'DNA_strand', 'TSS', 'Sigma', 'Con_level', 'Pro_seq']
        data_pd.to_excel(excel_writer=writer, sheet_name=sheet_name, index=False, header=header)
        writer.save()
        writer.close()
        print(sheet_name + ' to ' + xlsx_file)


write_data_dp_2_xlsx(dataset, xlsx_file, "dataset")
write_data_dp_2_xlsx(promoters_pd, xlsx_file, "promoter")
write_data_dp_2_xlsx(no_promoters_pd, xlsx_file, "no_promoter")
