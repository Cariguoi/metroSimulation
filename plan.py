import algo
import ligne
import station


class Plan:
    def __init__(self):
        self.budget = 0
        self.listStations = []
        self.listLines = []

    def createStation(self, x, y, name):
        self.listStations.append(station.Station(x, y, name))

    def createLine(self, name):
        self.listLines.append(ligne.Lines(name))

    #################
    def generateLine(self, listStationLine, Line):
        algo.list_coordinates_and_names(listStationLine)

    #################

    def addStationLine(self, listStationLine, Line):
        verif = False
        for i in self.listLines:
            if i.ligne == Line:
                i.listStation = listStationLine
                verif = True
                break
        if not verif:
            print("Ligne non existante")

