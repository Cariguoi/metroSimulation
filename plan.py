import algo
import ligne
import station
import map


class Plan:
    def __init__(self):
        self.budget = 0
        self.listStations = []
        self.listLines = []

    def createStation(self, name, x, y):
        self.listStations.append(station.Station(name, x, y))

    def createLine(self, name):
        self.listLines.append(ligne.Lines(name))

    def addStationLine(self, listStationLine, ligneName):
        verif = False
        for line in self.listLines:
            if line.name == ligneName:
                listopti = algo.list_coordinates_and_names(listStationLine)
                line.listStation = []
                for stationName in listopti:
                    line.listStation.append(station.searchStation(stationName, self.listStations))
                verif = True
                print(line.listStation)
                break
        if not verif:
            print("Ligne non existante")

    def showPlan(self):
        map.map(self.listStations, self.listLines)

