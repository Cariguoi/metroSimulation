import webbrowser

import folium


# lines = [line[stations]]
# ATTENTION : Ne fonctionne plus au dela de 5 ligne car manque de couleurs

def map(data, lines):
    map_osm = folium.Map(location=[48.87, 2.34], zoom_start=13)

    for i in data[:len(data)]:
        map_osm.add_child(folium.RegularPolygonMarker(location=[i[1], i[2]], popup=i[0],
                                                      fill_color='#132b5e', radius=5))

    colors = ["blue", "red", "orange", "green", "purple"]

    for line in lines:
        l = []
        for stations in line:
            x, y = stations.getCoordonate()
            l.append(tuple([x, y]))

        folium.PolyLine(l, color=colors[lines.index(line)], weight=2.5, opacity=1).add_to(map_osm)

    html_page = "map.html"
    map_osm.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)
