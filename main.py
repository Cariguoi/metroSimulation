import aws_connector
import plan
import map

if __name__ == '__main__':
    plan = plan.Plan()
    pricebykm = 75.000

    # get DATA [(name, x, y)]
    rds = aws_connector.RDS()
    data = rds.get_data()

    plan.createLine("1")
    for station in data:
        plan.createStation(station[0], station[1], station[2])

    map.map(data)

    #plan.addStationLine(data, "1")
