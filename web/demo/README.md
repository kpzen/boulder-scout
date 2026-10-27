# Boulder Scout demo

**Open the map: https://kpzen.github.io/boulder-scout-demo/**

Boulder Scout finds likely climbable boulders in Swedish forest from Lantmäteriet's LiDAR, so
climbers can scout new areas from home. This demo shows about 34,000 candidates over
3,300 km² of forest around Överturingen, central Sweden. It is a static site built by Boulder Scout's demo build; the code is at
https://github.com/kpzen/boulder-scout.

## Reading the map
- Each pin is a candidate boulder found in the LiDAR. Its colour is its score, how likely it is
  to be a boulder: red at the cutoff (2), yellow at 3.5, green from 5. Bigger pins are bigger
  boulders (plan-view area).
- The sliders hide candidates below a height (`max_h`, m), an area (`area_m2`, m²) or a score.
  Height and area start at 2 m and 8 m².
- Click a pin for its values, a **3D view** of its LiDAR points and its **Aerial photos**: a
  40 m colour crop, with ticks at the edges marking the candidate, and a link to Min karta.
- **Save** keeps a boulder in this browser only (blue pin); **Export as KML** downloads the saved
  ones, for Google My Maps or a GPS app.
- A pin is a guess, not a known boulder: some are vegetation, other terrain or man-made. Check
  the photo and the 3D view before you go.

## Credits
- LiDAR (Laserdata Skog) and ground model: © Lantmäteriet (CC BY 4.0).
- Aerial photos: Datakälla: Lantmäteriets Min karta, © Lantmäteriet, bearbetad, CC BY 4.0.
- Imagery basemap: Powered by Esri · Source: Esri, Vantor, GeoEye, Earthstar Geographics,
  CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community.
- Map basemap: © OpenStreetMap contributors.
