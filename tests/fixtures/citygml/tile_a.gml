<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
  xmlns:core="http://www.opengis.net/citygml/2.0"
  xmlns:gml="http://www.opengis.net/gml"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
  xmlns:xlink="http://www.w3.org/1999/xlink">
  <gml:boundedBy>
    <gml:Envelope srsName="urn:ogc:def:crs:EPSG::25833">
      <gml:lowerCorner>0 0</gml:lowerCorner>
      <gml:upperCorner>10 10</gml:upperCorner>
    </gml:Envelope>
  </gml:boundedBy>
  <core:cityObjectMember>
    <bldg:Building gml:id="building-1">
      <gml:description xlink:href="#surface-1" />
      <gml:surfaceMember gml:id="surface-1" />
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>
