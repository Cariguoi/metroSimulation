import algo
import algo2
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

    def addStationLines(self, listStationLines, lignesNames):
        verif = False
        i = 0
        linesIdx = [len(lignesNames)]
        for line in self.listLines:
            if line.name == lignesNames[i]:
                linesIdx.append(i)
                i = i + 1
            if i == len(lignesNames):
                linesOpti = algo2.list_coordinates_and_names(listStationLines, len(lignesNames))
                j = 0
                for lineOpti in linesOpti:
                    self.listLines[linesIdx[j]].listStation = []
                    for stationName in lineOpti:
                        self.listLines[linesIdx[j]].listStation.append(station.searchStation(stationName, self.listStations))
                        print(self.listLines[linesIdx[j]].listStation)
                    j = j + 1
                verif = True
                break
        if not verif:
            print("Au moins une ligne non existante")

    def showPlan(self):
        map.map(self.listStations, self.listLines)
