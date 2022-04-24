import plan


if __name__ == '__main__':
    plan = plan.Plan
    pricebykm = 75.000

    with open('file.csv') as csvDataFile:

        # read file as csv file
        csvReader = csv.reader(csvDataFile)

        # for every row, print the row
        for row in csvReader:
            print(row)

    for line in csv:
        plan.createStation()
