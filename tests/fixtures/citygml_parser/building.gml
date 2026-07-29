<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
  xmlns:core="http://www.opengis.net/citygml/2.0"
  xmlns:gml="http://www.opengis.net/gml"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0">
  <gml:boundedBy>
    <gml:Envelope srsName="urn:ogc:def:crs:EPSG::25833">
      <gml:lowerCorner>0 0 0</gml:lowerCorner>
      <gml:upperCorner>4 4 10</gml:upperCorner>
    </gml:Envelope>
  </gml:boundedBy>
  <core:cityObjectMember>
    <bldg:Building gml:id="building-parser-test">
      <bldg:boundedBy>
        <gml:Polygon>
          <gml:exterior>
            <gml:LinearRing>
              <gml:posList>0 0 0 4 0 0 4 4 0 0 4 0 0 0 0</gml:posList>
            </gml:LinearRing>
          </gml:exterior>
        </gml:Polygon>
      </bldg:boundedBy>
      <bldg:boundedBy>
        <gml:Polygon>
          <gml:exterior>
            <gml:LinearRing>
              <gml:posList>0 0 10 4 0 10 4 4 10 0 4 10 0 0 10</gml:posList>
            </gml:LinearRing>
          </gml:exterior>
        </gml:Polygon>
      </bldg:boundedBy>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>
