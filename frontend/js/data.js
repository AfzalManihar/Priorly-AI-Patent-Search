/* Demo data. Every patent here is fictional. Replace with your API responses later. */
window.CLAIM = "A system for wireless charging of an electric vehicle, comprising: a transmitter coil embedded in a road surface; a receiver coil mounted on an underside of the vehicle; and a controller configured to adjust a resonant frequency based on an alignment of the transmitter coil and the receiver coil.";

window.ELEMENTS = [
  { n: 1, text: "Transmitter coil embedded in a road surface", query: "road-embedded primary coil, in-pavement transmitter pad" },
  { n: 2, text: "Receiver coil mounted on the vehicle underside", query: "underbody pickup coil, chassis-mounted receiver" },
  { n: 3, text: "Controller adjusts resonant frequency from coil alignment", query: "frequency tuning on misalignment, adaptive resonance control" }
];

/* dense / bm25 = rank in each retriever. score = cross-encoder score after rerank. */
window.PATENTS = [
  { id: "US 9,104,382 B2", title: "Inductive power transfer system for roadway vehicles", assignee: "Voltaic Mobility Inc.", year: 2015, cpc: "B60L 53/12", el: [1, 2, 3], dense: 1, bm25: 4, score: 0.91,
    loc: "Col. 7, lines 12–30",
    passage: "A primary <mark data-e=\"1\">coil embedded beneath the road surface</mark> couples with a <mark data-e=\"2\">pickup coil fixed to the underside of the vehicle</mark>. A controller <mark data-e=\"3\">shifts the resonant frequency when lateral misalignment between the coils is detected</mark>.",
    why: "Discloses all three elements in one embodiment, including frequency adjustment on misalignment." },
  { id: "US 8,771,605 B1", title: "Dynamic charging pad with alignment sensing", assignee: "Gridline Systems", year: 2014, cpc: "B60L 53/38", el: [1, 2], dense: 2, bm25: 9, score: 0.84,
    loc: "Col. 4, lines 3–20",
    passage: "Charging pads are <mark data-e=\"1\">installed flush within the pavement</mark> and energize a <mark data-e=\"2\">receiver mounted beneath the chassis</mark>. Position sensors report lateral offset to the vehicle controller.",
    why: "Covers the road-side coil and the underbody receiver. Reports offset but never tunes frequency." },
  { id: "US 9,876,540 B2", title: "Coil misalignment compensation for inductive chargers", assignee: "Kestrel Power Labs", year: 2018, cpc: "H02J 50/90", el: [2, 3], dense: 6, bm25: 3, score: 0.78,
    loc: "Col. 9, lines 40–58",
    passage: "When the <mark data-e=\"2\">vehicle-side receiver coil</mark> drifts from nominal position, <mark data-e=\"3\">the controller retunes the operating frequency to restore the coupling coefficient</mark>.",
    why: "Strong on frequency retuning driven by receiver position. Road-embedded transmitter is not described." },
  { id: "US 8,904,567 B1", title: "Magnetic coupler tuning via capacitor bank switching", assignee: "Kestrel Power Labs", year: 2013, cpc: "H01F 38/14", el: [3], dense: 52, bm25: 2, score: 0.73,
    loc: "Col. 12, lines 5–22",
    passage: "A switched capacitor bank <mark data-e=\"3\">adjusts the resonant frequency of the primary circuit as the measured air gap and offset change</mark>.",
    why: "Frequency adjustment from offset, but described for generic couplers. Keyword search found it, dense search ranked it 52nd." },
  { id: "US 7,982,411 B2", title: "Resonant coupling frequency tuning circuit", assignee: "Kestrel Power Labs", year: 2011, cpc: "H02J 50/12", el: [3], dense: 38, bm25: 1, score: 0.71,
    loc: "Col. 5, lines 28–41",
    passage: "The controller <mark data-e=\"3\">adjusts the resonant frequency of the transmitter in response to a change in coupling coefficient</mark> measured at the receiver.",
    why: "Exact phrase match on resonant frequency adjustment. Ranked 1st by keyword search, 38th by dense search." },
  { id: "US 8,332,104 B2", title: "Electrified roadway with segmented transmitter coils", assignee: "Aurelia Transport Tech", year: 2012, cpc: "B60L 5/00", el: [1], dense: 5, bm25: 8, score: 0.68,
    loc: "Col. 3, lines 50–67",
    passage: "A series of <mark data-e=\"1\">transmitter coils buried under the asphalt layer</mark> are energized sequentially as a vehicle passes.",
    why: "Road-embedded transmitter coils only. No receiver or tuning detail." },
  { id: "US 10,010,277 B2", title: "In-road wireless power module with sealed housing", assignee: "Gridline Systems", year: 2018, cpc: "E01C 11/00", el: [1], dense: 4, bm25: 31, score: 0.66,
    loc: "Col. 2, lines 8–19",
    passage: "The module houses a <mark data-e=\"1\">transmitter coil set into a recess in the road surface</mark> and sealed against moisture.",
    why: "Describes the road-side hardware in detail. Nothing on the vehicle side." },
  { id: "US 9,230,771 B1", title: "Adaptive impedance matching for wireless chargers", assignee: "Lumen Drive Corp.", year: 2016, cpc: "H02J 50/40", el: [3], dense: 9, bm25: 5, score: 0.64,
    loc: "Col. 6, lines 14–27",
    passage: "<mark data-e=\"3\">The matching network adapts so that the system remains at resonance</mark> as coil spacing varies.",
    why: "Adaptive resonance, but through impedance matching rather than frequency control." },
  { id: "US 9,450,019 B2", title: "Vehicle underbody receiver assembly", assignee: "Nordhaven Automotive", year: 2016, cpc: "B60L 53/12", el: [2], dense: 3, bm25: 17, score: 0.62,
    loc: "Col. 5, lines 1–16",
    passage: "A <mark data-e=\"2\">receiver coil is bolted to the underside of the vehicle floor pan</mark> behind a non-metallic shield.",
    why: "Clear match for the underbody receiver. Other elements are absent." },
  { id: "US 7,455,912 B2", title: "Contactless energy supply for rail vehicles", assignee: "Brenner Elektro", year: 2008, cpc: "B60M 7/00", el: [1, 2], dense: 7, bm25: 24, score: 0.55,
    loc: "Col. 3, lines 9–24",
    passage: "<mark data-e=\"1\">A conductor loop laid between the rails</mark> induces current in a <mark data-e=\"2\">pickup mounted under the carriage</mark>.",
    why: "Same structure for rail vehicles. Weaker because it is not a road vehicle and has no tuning." },
  { id: "US 10,305,688 B2", title: "Foreign object detection in charging pads", assignee: "Gridline Systems", year: 2019, cpc: "H02J 50/60", el: [1], dense: 8, bm25: 40, score: 0.41,
    loc: "Col. 8, lines 2–11",
    passage: "Sensing coils in the <mark data-e=\"1\">transmitter pad</mark> detect metallic objects before power is applied.",
    why: "Mentions the transmitter pad only as context for object detection." },
  { id: "US 9,611,223 B2", title: "Portable receiver coil for electric bicycles", assignee: "Pedalwatt Ltd.", year: 2017, cpc: "B62J 43/00", el: [2], dense: 10, bm25: 55, score: 0.38,
    loc: "Col. 4, lines 30–39",
    passage: "A <mark data-e=\"2\">receiver coil in the bicycle frame</mark> accepts power from a docking station.",
    why: "Receiver coil on a bicycle. Different vehicle class." }
];

window.EVAL = {
  stages: [
    { key: "dense", name: "Dense only", recall: 0.41, precision: 0.22, mrr: 0.35 },
    { key: "hybrid", name: "+ BM25 and RRF", recall: 0.58, precision: 0.29, mrr: 0.46 },
    { key: "rerank", name: "+ Cross-encoder rerank", recall: 0.72, precision: 0.38, mrr: 0.58 },
    { key: "full", name: "+ Decompose and HyDE", recall: 0.81, precision: 0.44, mrr: 0.66 }
  ],
  byType: [
    { type: "Mechanical claims", n: 74, dense: 0.47, hybrid: 0.60, full: 0.83 },
    { type: "Chemical and formula queries", n: 52, dense: 0.29, hybrid: 0.57, full: 0.78 },
    { type: "Software and protocol claims", n: 48, dense: 0.50, hybrid: 0.58, full: 0.84 },
    { type: "Queries containing an exact patent ID", n: 26, dense: 0.12, hybrid: 0.69, full: 0.88 }
  ]
};

window.SAVED_SEED = [
  { title: "Inductive roadway charging, claim 1", date: "28 Oct 2026", n: 10, best: 0.91 },
  { title: "Solid-state electrolyte coating, claims 1-4", date: "21 Oct 2026", n: 12, best: 0.87 },
  { title: "Edge caching of encrypted media segments", date: "14 Oct 2026", n: 9, best: 0.79 }
];
