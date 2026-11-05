"""Generate the bundled SAMPLE corpus and query set.

Everything here is fictional: patent numbers, companies and passages are invented so the whole
stack (API, UI, evaluation) runs out of the box. Replace it with real PatentsView data
(see README, "Using real data"). Numbers measured on this sample say nothing about real quality.

Usage:  python -m ingest.make_sample_data
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"


def P(pid, title, assignee, year, cpc, abstract, claim, paras):
    """Build one patent record. Patents whose id starts with US 11, are query-only (see QUERY_ONLY)."""
    numbered = []
    for i, item in enumerate(paras, start=1):
        no, text = item if isinstance(item, tuple) else (f"{i:04d}", item)
        numbered.append({"no": no, "text": text})
    return {"id": pid, "title": title, "assignee": assignee, "year": year, "cpc": cpc,
            "abstract": abstract, "claims": [claim], "paragraphs": numbered}


PATENTS = [
    # ---- wireless charging cluster ------------------------------------------------
    P("US 9,104,382 B2", "Inductive power transfer system for roadway vehicles", "Voltaic Mobility Inc.", 2015, "B60L 53/12",
      "A wireless power system for electric vehicles transfers energy from a primary coil in a roadway to a pickup coil on the vehicle, with frequency control to maintain coupling.",
      "A roadway power transfer system comprising: a primary coil embedded beneath a road surface; a pickup coil fixed to an underside of a vehicle; and a controller that shifts a resonant frequency when lateral misalignment between the coils is detected.",
      [("0011", "Electric vehicles benefit from charging while parked or in motion, which reduces battery size and the number of charging stops."),
       ("0042", "A primary coil embedded beneath the road surface couples with a pickup coil fixed to the underside of the vehicle. A controller shifts the resonant frequency when lateral misalignment between the coils is detected."),
       ("0043", "The controller measures the coupling coefficient and selects a drive frequency from a stored table so that power transfer efficiency stays above a target during misalignment.")]),
    P("US 8,771,605 B1", "Dynamic charging pad with alignment sensing", "Gridline Systems", 2014, "B60L 53/38",
      "Charging pads installed in pavement energize a vehicle receiver while position sensors report offset.",
      "A charging pad installed flush within pavement and a receiver mounted beneath a vehicle chassis, wherein position sensors report lateral offset to a vehicle controller.",
      [("0008", "Parking structures and roads may include charging pads that energize vehicles without plug connections."),
       ("0021", "Charging pads are installed flush within the pavement and energize a receiver mounted beneath the chassis. Position sensors report lateral offset to the vehicle controller."),
       ("0022", "The offset report is displayed to the driver to guide the vehicle toward the centre of the pad.")]),
    P("US 9,876,540 B2", "Coil misalignment compensation for inductive chargers", "Kestrel Power Labs", 2018, "H02J 50/90",
      "Compensates for receiver coil drift by retuning the operating frequency to restore the coupling coefficient.",
      "A method comprising: detecting drift of a vehicle-side receiver coil; and retuning an operating frequency to restore a coupling coefficient.",
      [("0015", "Inductive chargers lose efficiency when coils drift from their nominal positions. The resonant behaviour of the circuit is similar to that described in US 7,982,411 B2."),
       ("0063", "When the vehicle-side receiver coil drifts from nominal position, the controller retunes the operating frequency to restore the coupling coefficient."),
       ("0064", "A lookup table maps measured drift to a frequency offset.")]),
    P("US 8,904,567 B1", "Magnetic coupler tuning via capacitor bank switching", "Kestrel Power Labs", 2013, "H01F 38/14",
      "A switched capacitor bank retunes a magnetic coupler as air gap and offset change.",
      "A magnetic coupler comprising a switched capacitor bank that adjusts a resonant frequency of a primary circuit as a measured air gap and offset change.",
      [("0009", "Magnetic couplers transfer power across an air gap and are sensitive to alignment. Related tuning methods are known from US 7,982,411 B2."),
       ("0072", "A switched capacitor bank adjusts the resonant frequency of the primary circuit as the measured air gap and offset change."),
       ("0073", "Capacitors are switched by semiconductor relays under control of a microcontroller.")]),
    P("US 7,982,411 B2", "Resonant coupling frequency tuning circuit", "Kestrel Power Labs", 2011, "H02J 50/12",
      "A circuit that tunes a transmitter to remain at resonance as coupling varies.",
      "A circuit comprising a controller that adjusts a resonant frequency of a transmitter in response to a change in coupling coefficient measured at a receiver.",
      [("0007", "Resonant inductive links require tuning because the coupling coefficient varies with distance."),
       ("0034", "The controller adjusts the resonant frequency of the transmitter in response to a change in coupling coefficient measured at the receiver."),
       ("0035", "Measurements are reported over a low power radio link.")]),
    P("US 8,332,104 B2", "Electrified roadway with segmented transmitter coils", "Aurelia Transport Tech", 2012, "B60L 5/00",
      "A roadway carries segmented transmitter coils that are energized as a vehicle passes.",
      "A roadway comprising a series of transmitter coils buried under an asphalt layer and energized sequentially.",
      [("0019", "A series of transmitter coils buried under the asphalt layer are energized sequentially as a vehicle passes."),
       ("0020", "Segment switches connect each coil to a supply cable.")]),
    P("US 10,010,277 B2", "In-road wireless power module with sealed housing", "Gridline Systems", 2018, "E01C 11/00",
      "A sealed module houses a transmitter coil in a recess of the road surface.",
      "A module comprising a sealed housing and a transmitter coil set into a recess in a road surface.",
      [("0012", "The module houses a transmitter coil set into a recess in the road surface and sealed against moisture."),
       ("0013", "A drainage channel carries water away from the housing.")]),
    P("US 9,230,771 B1", "Adaptive impedance matching for wireless chargers", "Lumen Drive Corp.", 2016, "H02J 50/40",
      "A matching network keeps a wireless charger at resonance as coil spacing changes.",
      "A charger comprising a matching network that adapts so the system remains at resonance as coil spacing varies.",
      [("0026", "The matching network adapts so that the system remains at resonance as coil spacing varies."),
       ("0027", "Variable capacitors are driven by a closed loop controller.")]),
    P("US 9,450,019 B2", "Vehicle underbody receiver assembly", "Nordhaven Automotive", 2016, "B60L 53/12",
      "A receiver coil assembly mounts to the vehicle floor pan behind a shield.",
      "A receiver assembly comprising a receiver coil bolted to an underside of a vehicle floor pan behind a non-metallic shield.",
      [("0014", "A receiver coil is bolted to the underside of the vehicle floor pan behind a non-metallic shield."),
       ("0015", "The shield limits stray magnetic fields inside the cabin.")]),
    P("US 7,455,912 B2", "Contactless energy supply for rail vehicles", "Brenner Elektro", 2008, "B60M 7/00",
      "A conductor loop between the rails induces current in a pickup under a carriage.",
      "A rail system comprising a conductor loop laid between rails and a pickup mounted under a carriage.",
      [("0010", "A conductor loop laid between the rails induces current in a pickup mounted under the carriage."),
       ("0011", "Trains draw traction power without overhead lines.")]),
    P("US 10,305,688 B2", "Foreign object detection in charging pads", "Gridline Systems", 2019, "H02J 50/60",
      "Sensing coils in a transmitter pad detect metallic objects before power is applied.",
      "A pad comprising sensing coils that detect metallic objects before power is applied.",
      [("0018", "Sensing coils in the transmitter pad detect metallic objects before power is applied."),
       ("0019", "Detection thresholds are calibrated in the factory.")]),
    P("US 9,611,223 B2", "Portable receiver coil for electric bicycles", "Pedalwatt Ltd.", 2017, "B62J 43/00",
      "A bicycle frame carries a receiver coil that accepts power from a docking station.",
      "A bicycle comprising a receiver coil in a frame that accepts power from a docking station.",
      [("0016", "A receiver coil in the bicycle frame accepts power from a docking station."),
       ("0017", "Docking stations are installed at public racks.")]),
    P("US 11,200,001 B2", "Wireless electric vehicle charging system", "Query Corp.", 2020, "B60L 53/12",
      "Wireless charging of an electric vehicle with road coils and frequency adjustment.",
      "A system for wireless charging of an electric vehicle, comprising: a transmitter coil embedded in a road surface; a receiver coil mounted on an underside of the vehicle; and a controller configured to adjust a resonant frequency based on an alignment of the transmitter coil and the receiver coil.",
      [("0001", "This patent describes wireless charging of electric vehicles.")]),
    P("US 11,200,002 B2", "Frequency tuning for wireless chargers", "Query Corp.", 2020, "H02J 50/12",
      "Tuning a wireless charger from a measured coupling coefficient.",
      "A method of tuning a wireless charger, the method improving on the circuit of US 7,982,411 B2, comprising: measuring a coupling coefficient at a receiver; and adjusting a resonant frequency of a transmitter based on the coupling coefficient.",
      [("0001", "This patent describes a tuning method for wireless chargers.")]),
    # ---- solid-state battery cluster -----------------------------------------------
    P("US 9,500,101 B2", "Tantalum-doped garnet solid electrolyte", "Helio Cell", 2016, "H01M 10/0562",
      "A garnet electrolyte doped with tantalum for improved lithium ion conductivity.",
      "A solid electrolyte comprising Li7La3Zr2O12 in which zirconium is partly substituted by tantalum.",
      [("0021", "The electrolyte comprises Li7La3Zr2O12 doped with tantalum on the zirconium site, which stabilises the cubic phase and raises ionic conductivity."),
       ("0022", "Doping levels between 0.25 and 0.5 tantalum per formula unit are preferred.")]),
    P("US 9,720,330 B1", "Alumina interlayer for lithium garnet electrolytes", "Helio Cell", 2017, "H01M 10/0525",
      "An Al2O3 coating on garnet grain boundaries lowers interfacial resistance against lithium metal.",
      "An electrolyte comprising a garnet ceramic having an Al2O3 coating on grain boundaries.",
      [("0017", "A coating of Al2O3 on the grain boundaries of the garnet electrolyte reduces interfacial resistance against lithium metal."),
       ("0018", "The coating is applied by atomic layer deposition.")]),
    P("US 8,615,220 B2", "Lithium iron phosphate cathode with carbon coating", "Ferron Energy", 2013, "H01M 4/58",
      "LiFePO4 cathode particles carry a conductive carbon coating.",
      "A cathode comprising LiFePO4 particles having a conductive carbon coating.",
      [("0013", "LiFePO4 particles carry a conductive carbon coating deposited by pyrolysis of sucrose, which improves electronic conductivity."),
       ("0014", "Coating thickness is between two and ten nanometres.")]),
    P("US 9,010,874 B2", "Sulfide solid electrolyte with argyrodite structure", "Nordcell", 2015, "H01M 10/0562",
      "A sulfide electrolyte with argyrodite structure prepared by ball milling.",
      "A solid electrolyte comprising Li6PS5Cl having an argyrodite crystal structure.",
      [("0020", "The electrolyte Li6PS5Cl is formed by ball milling lithium sulfide, phosphorus pentasulfide and lithium chloride."),
       ("0021", "Annealing at 550 degrees Celsius produces the argyrodite phase.")]),
    P("US 8,992,555 B1", "Conductive carbon layers on olivine cathode particles", "Ferron Energy", 2014, "H01M 4/36",
      "Olivine phosphate cathode particles coated with a conductive carbon layer.",
      "A cathode material comprising olivine phosphate particles coated with a conductive carbon layer.",
      [("0015", "Olivine phosphate cathode particles are coated with a thin conductive carbon layer to reduce resistance."),
       ("0016", "The carbon layer is derived from an organic precursor heated under inert gas.")]),
    P("US 9,300,909 B2", "Protective layer for lithium metal anodes", "Helio Cell", 2015, "H01M 4/134",
      "A polymer layer suppresses dendrite growth on lithium metal anodes.",
      "An anode comprising lithium metal and a polymer protective layer.",
      [("0012", "A polymer protective layer suppresses dendrite growth on lithium metal anodes."),
       ("0013", "The layer is permeable to lithium ions.")]),
    P("US 9,880,421 B2", "Hot pressing and sintering of garnet pellets", "Kilnworks", 2017, "C04B 35/50",
      "Garnet pellets are sintered in alumina crucibles, introducing aluminium at grain boundaries.",
      "A method comprising sintering pellets of Li7La3Zr2O12 at a temperature above one thousand degrees Celsius.",
      [("0024", "Pellets of Li7La3Zr2O12 are sintered at 1100 degrees Celsius in an alumina crucible, and aluminium diffusion leaves Al2O3 at the grain boundaries."),
       ("0025", "Cooling is performed at a controlled rate.")]),
    P("US 11,300,001 B2", "Garnet electrolyte with grain boundary coating", "Query Corp.", 2020, "H01M 10/0562",
      "A tantalum-doped garnet electrolyte with an alumina grain boundary coating.",
      "A solid electrolyte comprising Li7La3Zr2O12 doped with tantalum, and an Al2O3 coating on grain boundaries.",
      [("0001", "This patent describes a garnet solid electrolyte.")]),
    P("US 11,300,002 B2", "Carbon coated olivine cathode", "Query Corp.", 2020, "H01M 4/58",
      "A carbon coated lithium iron phosphate cathode.",
      "A cathode comprising LiFePO4 particles having a conductive carbon coating.",
      [("0001", "This patent describes an olivine cathode.")]),
    # ---- edge caching cluster ------------------------------------------------------
    P("US 9,100,201 B2", "Edge cache for encrypted media segments", "Streamline Networks", 2016, "H04N 21/231",
      "An edge node stores encrypted media segments and serves them without decryption.",
      "A method comprising storing encrypted media segments at an edge node and serving a segment without decrypting it.",
      [("0014", "The edge node stores encrypted media segments and serves them without decrypting them, so keys never reach the edge."),
       ("0015", "Segments are indexed by identifier and byte range.")]),
    P("US 9,400,330 B1", "Prefetching media segments based on viewing prediction", "Streamline Networks", 2017, "H04L 67/568",
      "A prediction module estimates future viewing and a prefetch controller loads segments before requests.",
      "A system comprising a prediction module that estimates future viewing of media segments and a prefetch controller that retrieves predicted segments into a cache.",
      [("0022", "A prediction module estimates future viewing of media segments, and a prefetch controller retrieves the predicted segments into the cache before requests arrive."),
       ("0023", "Predictions are refreshed every few seconds from playback telemetry.")]),
    P("US 8,900,118 B2", "Content-addressed storage for CDN nodes", "Cachewell", 2014, "H04L 67/1097",
      "Segments are stored under a hash of their content so duplicates are stored once.",
      "A node comprising storage that keys media segments by a hash of their content.",
      [("0011", "Segments are stored under a hash of their content so that duplicate segments are stored once."),
       ("0012", "Lookups use the hash taken from a manifest.")]),
    P("US 9,250,764 B2", "Token authorization of cached media segments", "Streamline Networks", 2015, "H04L 63/08",
      "An edge node validates an authorization token before serving a cached segment.",
      "A method comprising receiving a segment request with a token and validating the token before serving a cached segment.",
      [("0016", "The edge node validates an authorization token for each segment request before serving the cached segment."),
       ("0017", "Tokens expire after a configured period.")]),
    P("US 9,600,872 B1", "Adaptive bitrate selection at the edge", "Cachewell", 2016, "H04N 21/2662",
      "An edge node selects a bitrate representation from measured throughput.",
      "A method comprising measuring throughput at an edge node and selecting a bitrate representation.",
      [("0018", "The edge node selects a bitrate representation from measured throughput."),
       ("0019", "Representations are listed in a manifest.")]),
    P("US 9,770,515 B2", "Cache eviction using popularity prediction", "Cachewell", 2017, "G06F 12/123",
      "A popularity model predicts demand and evicts the least requested segments.",
      "A cache comprising a popularity model that predicts requests and evicts segments with the lowest predicted demand.",
      [("0020", "A popularity model predicts which segments will be requested again, and evicts the segments with the lowest predicted demand."),
       ("0021", "The model is trained on request logs.")]),
    P("US 11,400,001 B2", "Authorized serving of encrypted segments from an edge node", "Query Corp.", 2020, "H04L 63/08",
      "Serving encrypted segments from an edge node with token checks.",
      "A method comprising: storing encrypted media segments at an edge node; validating an authorization token for each segment request; and serving the segment without decrypting at the edge node.",
      [("0001", "This patent describes secure edge delivery.")]),
    P("US 11,400,002 B2", "Predictive prefetch for streaming caches", "Query Corp.", 2020, "H04L 67/568",
      "A prediction module with a prefetch controller for an edge cache.",
      "A system comprising: a prediction module that estimates future viewing of media segments; and a prefetch controller that retrieves predicted segments into an edge cache before they are requested.",
      [("0001", "This patent describes predictive prefetch.")]),
    # ---- distractors ---------------------------------------------------------------
    P("US 8,100,501 B2", "Robotic gripper with compliant fingers", "Handwerk Robotics", 2012, "B25J 15/00",
      "A gripper with compliant fingers adapts to object shape.",
      "A gripper comprising compliant fingers and a controller that adjusts grip force based on a sensor.",
      [("0010", "Compliant fingers conform to objects of varied shape. A controller adjusts grip force based on a tactile sensor.")]),
    P("US 8,300,602 B2", "Self-expanding vascular stent", "Medivale", 2013, "A61F 2/90",
      "A stent that expands in a blood vessel.",
      "A stent comprising a nitinol mesh that expands at body temperature.",
      [("0009", "The nitinol mesh expands at body temperature and presses against the vessel wall.")]),
    P("US 9,000,703 B1", "Voice coil autofocus actuator", "Lumina Optics", 2014, "G02B 7/09",
      "An autofocus actuator for camera modules driven by a voice coil.",
      "An actuator comprising a voice coil and a controller that adjusts a drive current based on lens position.",
      [("0012", "A voice coil moves the lens and a controller adjusts the drive current based on the lens position reported by a hall sensor.")]),
    P("US 8,700,804 B2", "Soil moisture sensor for irrigation", "Agrisense", 2012, "A01G 25/16",
      "A sensor measures soil moisture to control irrigation valves.",
      "A system comprising a soil moisture sensor and a valve controller.",
      [("0008", "The controller opens the valve when the moisture measured at the sensor falls below a threshold.")]),
    P("US 9,200,905 B2", "Wind turbine blade pitch controller", "Aeolus Power", 2015, "F03D 7/02",
      "A controller adjusts blade pitch to limit loads.",
      "A wind turbine comprising a controller that adjusts blade pitch based on measured wind speed.",
      [("0014", "The controller adjusts blade pitch based on wind speed and rotor frequency to limit mechanical loads.")]),
    P("US 7,900,106 B2", "Lightweight bicycle frame", "Pedalwatt Ltd.", 2010, "B62K 19/02",
      "A bicycle frame made from tubular aluminium.",
      "A bicycle frame comprising hydroformed aluminium tubes joined at a head tube.",
      [("0006", "Hydroformed aluminium tubes are welded to the head tube and the bottom bracket shell.")]),
]

QUERIES = [
    {"id": "q1", "patent_id": "US 11,200,001 B2", "year": 2020, "type": "mechanical", "split": "dev",
     "relevant": ["US 9,104,382 B2", "US 8,771,605 B1", "US 9,876,540 B2", "US 8,904,567 B1", "US 7,982,411 B2"]},
    {"id": "q2", "patent_id": "US 11,200,002 B2", "year": 2020, "type": "exact_id", "split": "dev",
     "relevant": ["US 7,982,411 B2", "US 9,876,540 B2", "US 8,904,567 B1"]},
    {"id": "q3", "patent_id": "US 11,300,001 B2", "year": 2020, "type": "chemical", "split": "val",
     "relevant": ["US 9,500,101 B2", "US 9,720,330 B1", "US 9,880,421 B2"]},
    {"id": "q4", "patent_id": "US 11,300,002 B2", "year": 2020, "type": "chemical", "split": "val",
     "relevant": ["US 8,615,220 B2", "US 8,992,555 B1"]},
    {"id": "q5", "patent_id": "US 11,400,001 B2", "year": 2020, "type": "software", "split": "test",
     "relevant": ["US 9,100,201 B2", "US 9,250,764 B2", "US 8,900,118 B2"]},
    {"id": "q6", "patent_id": "US 11,400,002 B2", "year": 2020, "type": "software", "split": "test",
     "relevant": ["US 9,400,330 B1", "US 9,770,515 B2"]},
]


QUERY_ONLY = {q["patent_id"] for q in QUERIES}


def main() -> None:
    """Query patents supply claims for evaluation but are NOT searchable (so a pasted claim never finds itself)."""
    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {p["id"]: p for p in PATENTS}
    searchable = [p for p in PATENTS if p["id"] not in QUERY_ONLY]
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for patent in searchable:
            handle.write(json.dumps(patent) + "\n")
    queries = []
    for q in QUERIES:
        queries.append({**q, "claim": by_id[q["patent_id"]]["claims"][0]})
    with open(OUT / "queries.json", "w", encoding="utf-8") as handle:
        json.dump({"dataset_version": "sample-1", "sample": True, "queries": queries}, handle, indent=2)
    print(f"wrote {len(searchable)} searchable patents and {len(queries)} queries to {OUT}")


if __name__ == "__main__":
    main()
