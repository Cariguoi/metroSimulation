import aws_connector
import plan
import map

if __name__ == '__main__':
    plan = plan.Plan()
    #pricebykm = 75.000

    # get DATA [(name, x, y)]
    #rds = aws_connector.RDS()
    #data = rds.get_data()
    data = [("Varenne", 48.856883, 2.315440), ("Solférino", 48.858207, 2.323406), ("Sevres - Babylone", 48.851468, 2.326531), ("Mabillon", 48.85291050815073, 2.33571101511061), ("Saint-Germain-des-Prés", 48.85364019156512, 2.3337395326064563), ("Assemblée Nationale", 48.86121153712306, 2.320341664343573), ("Pont de l'Alma", 48.862150346680146, 2.300766611362447), ("Cambronne", 48.8476376489699, 2.301200922831332), ("Volontaires", 48.841533302783304, 2.3082740002897872), ("Duroc", 48.84704562110988, 2.3162157013047957)]

    plan.createLine("1")
    plan.createLine("2")
    plan.createLine("3")
    for station in data:
        plan.createStation(station[0], station[1], station[2])
    #plan.addStationLine(data, "1")
    plan.addStationLines(data, ["1", "2", "3"])
    plan.showPlan()