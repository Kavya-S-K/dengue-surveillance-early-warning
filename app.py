
from pathlib import Path
import io, json, re, zipfile, tempfile, warnings
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
DATE, LOCALITY, TARGET = "week_start", "locality", "dengue_cases_next_week"
COLORS = {"Low":"#4ba866","Moderate":"#f2c94c","High":"#f2994a","Very High":"#d93636"}
DENGUE_LOCALITIES = ["Balaramapuram","Pallichal","Maranalloor","Malayinkeezhu","Kalliyoor",
                     "Vilappil","Vilavoorkal","Peringamala","Karode","Vellayani"]
SENSOR_MAP_DEFAULT = {
    "Bhagavathynada":"Pallichal","Maranalloor":"Maranalloor","Puliyarakonam":"Vilappil",
    "Ooruttambalam":"Malayinkeezhu","Machel":"Malayinkeezhu","Balaramapuram":"Balaramapuram",
    "Pallichal":"Pallichal","Vellayani":"Vellayani","Pravachambalam":"Pallichal",
    "Malayam":"Vilavoorkal","Malayinkeezhu":"Malayinkeezhu","Vilappilsala":"Vilappil",
    "Peyad":"Vilappil","Naruvamoodu":"Pallichal"
}
FEATURES = [
    "dengue_cases_current_week","dengue_lag1","dengue_lag2","dengue_lag3","dengue_lag4",
    "iot_temperature_mean","iot_temperature_mean_lag1","iot_humidity_mean","iot_humidity_mean_lag1",
    "iot_rainfall_weekly","iot_rainfall_weekly_lag1","iot_wind_speed_mean","iot_wind_speed_mean_lag1",
    "week_sin","week_cos"
]

st.set_page_config(page_title="Nemom Dengue Early Warning Dashboard", page_icon="🦟", layout="wide")
st.markdown("""<style>
/* ---------- Overall layout ---------- */
.block-container{padding-top:1.45rem;padding-bottom:3rem;max-width:1450px}
[data-testid="stMetricValue"]{font-size:1.45rem;font-weight:700}
[data-testid="stMetricLabel"]{font-size:.82rem}
[data-testid="stMetricDelta"]{font-size:.78rem}

/* ---------- Main dashboard banner ---------- */
.dashboard-hero{
    position:relative;
    padding:2.15rem 2.4rem 2rem;
    border:1px solid rgba(148,163,184,.34);
    border-radius:18px;
    background:
        radial-gradient(circle at 50% 0%, rgba(255,255,255,.98), rgba(255,255,255,0) 55%),
        linear-gradient(135deg,rgba(240,248,250,.99),rgba(225,239,247,.99));
    margin:0 0 1.8rem 0;
    box-shadow:0 8px 26px rgba(15,23,42,.16);
    text-align:center;
    overflow:hidden;
}
.dashboard-hero::before{
    content:"";
    display:block;
    width:58px;
    height:4px;
    margin:0 auto .95rem;
    border-radius:99px;
    background:#2b6f9f;
}
.dashboard-hero h1{
    margin:0;
    font-size:2.25rem;
    line-height:1.15;
    font-weight:800;
    color:#123e60;
    letter-spacing:-.025em;
}
.dashboard-hero p{
    max-width:820px;
    margin:.72rem auto 0;
    color:#536575;
    font-size:.96rem;
    line-height:1.55;
}

/* ---------- Section hierarchy ---------- */
.section-heading{
    margin-top:.15rem;
    margin-bottom:.2rem;
    font-size:1.16rem;
    font-weight:700;
    color:#e8eef5;
}
.section-subtitle{
    color:#8fa0b3;
    font-size:.82rem;
    line-height:1.45;
    margin-bottom:.85rem;
}
.section-note{
    color:#8fa0b3;
    font-size:.8rem;
    line-height:1.45;
    margin:.15rem 0 .8rem;
}
.card-label{
    color:#8fa0b3;
    font-size:.72rem;
    text-transform:uppercase;
    letter-spacing:.06em;
    font-weight:700;
}
.card-value{
    color:#f3f6fa;
    font-size:1.18rem;
    font-weight:700;
    margin-top:.12rem;
}
.card-note{
    color:#7f91a5;
    font-size:.73rem;
    line-height:1.35;
    margin-top:.25rem;
}

/* ---------- KPI cards ---------- */
.kpi-grid{
    margin:.4rem 0 1.25rem;
}
.kpi-card{
    min-height:105px;
    padding:.85rem .95rem;
    border:1px solid rgba(148,163,184,.18);
    border-radius:12px;
    background:linear-gradient(180deg,rgba(24,29,39,.92),rgba(20,25,34,.92));
    box-shadow:0 3px 12px rgba(0,0,0,.13);
}
.kpi-card .stMetric{padding:0}
.kpi-card [data-testid="stMetricLabel"]{white-space:normal!important;line-height:1.25!important;overflow-wrap:anywhere!important}
.kpi-card [data-testid="stMetricValue"]{white-space:normal!important;line-height:1.15!important;overflow-wrap:anywhere!important;font-size:1.15rem!important}

/* ---------- Forecast control card ---------- */
.control-card{
    padding:1rem 1.05rem .9rem;
    border:1px solid rgba(148,163,184,.18);
    border-radius:12px;
    background:rgba(22,27,36,.72);
    margin-bottom:1.2rem;
}

/* ---------- Selected locality panel ---------- */
.locality-card{
    padding:.95rem 1rem 1rem;
    border:1px solid rgba(148,163,184,.18);
    border-radius:12px;
    background:rgba(22,27,36,.72);
    min-height:100%;
}
.locality-card-title{
    font-size:1.05rem;
    font-weight:700;
    color:#eef4f8;
    margin-bottom:.15rem;
}

/* ---------- Tables ---------- */
[data-testid="stDataFrame"]{
    border:1px solid rgba(148,163,184,.15);
    border-radius:10px;
    overflow:hidden;
}

/* ---------- Spacing ---------- */
hr{margin:1.25rem 0!important;border-color:rgba(148,163,184,.12)!important}
.stButton>button{border-radius:9px}
</style>""", unsafe_allow_html=True)

def read_csv_bytes(b, name="file.csv"):
    try:
        return pd.read_csv(io.BytesIO(b), low_memory=False)
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(b), encoding="latin1", low_memory=False)
        except Exception as e:
            raise ValueError(f"Could not read {name} as CSV: {e}")

def norm_name(x):
    return re.sub(r"[^a-z0-9]+","_",str(x).strip().lower()).strip("_")

def find_col(df, candidates):
    nmap={norm_name(c):c for c in df.columns}
    for c in candidates:
        if norm_name(c) in nmap: return nmap[norm_name(c)]
    for c in df.columns:
        n=norm_name(c)
        if any(k in n for k in candidates): return c
    return None

def parse_date_col(df):
    candidates=["date","datetime","timestamp","time","record_date","report_date","observation_date"]
    c=find_col(df,candidates)
    if c is None:
        for c0 in df.columns:
            if df[c0].dtype=="object":
                p=pd.to_datetime(df[c0],errors="coerce")
                if p.notna().mean()>.65: c=c0; break
    if c is None: return None
    return pd.to_datetime(df[c],errors="coerce")

def canonical_locality(s):
    if pd.isna(s): return np.nan
    x=str(s).strip().lower()
    aliases={
        "balaramapuram":"Balaramapuram","pallichal":"Pallichal","maranalloor":"Maranalloor",
        "malayinkeezhu":"Malayinkeezhu","kalliyoor":"Kalliyoor","vilappil":"Vilappil",
        "vilavoorkal":"Vilavoorkal","peringamala":"Peringamala","karode":"Karode","vellayani":"Vellayani"
    }
    for k,v in aliases.items():
        if k in x: return v
    return str(s).strip()

def extract_health_from_csv(files):
    frames=[]
    for name,b in files:
        d=read_csv_bytes(b,name)
        dt=parse_date_col(d)
        loc=find_col(d,["locality","place","location","panchayat","lsgd","village","area"])
        case=find_col(d,["dengue_cases","dengue_case","cases","dengue","positive","count","number"])
        if dt is None or loc is None or case is None:
            continue
        x=pd.DataFrame({"date":dt,"locality":d[loc].map(canonical_locality),
                        "dengue_cases":pd.to_numeric(d[case],errors="coerce").fillna(0)})
        x=x[x["date"].notna() & x["locality"].notna()]
        frames.append(x)
    if not frames: raise ValueError("No uploaded health CSV matched date + locality + dengue case fields.")
    x=pd.concat(frames,ignore_index=True)
    x=x[x.locality.isin(DENGUE_LOCALITIES)]
    # Treat rows as observations; aggregate counts when multiple records occur on a date/locality.
    return x.groupby(["date","locality"],as_index=False)["dengue_cases"].sum()

def extract_health_from_pdf(files, progress=None):
    """Parse the original Kerala IDSP daily-report layout.

    The dengue section on page 2 has rows such as:
    TVM: 3  Pangappara, Puthenthope, Andoorkonam
    We use the TVM row only and convert each listed Nemom locality into a
    daily locality report event. Explicit numeric suffixes (e.g. "Locality 2")
    are honored. This is intentionally different from the processed Health
    Summary PDF, which is not required by this application.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ValueError("PDF health-data support needs pdfplumber. Run: pip install -r requirements.txt")

    rows=[]
    report_dates=[]
    reports_seen=0
    dengue_sections=0
    report_patters=[
        re.compile(r"(?:dtd|dated)\s*(\d{1,2}[./-]\d{1,2}[./-]20\d{2})",re.I),
        re.compile(r"daily report on\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?([A-Za-z]+\s+\d{1,2},\s+20\d{2})",re.I),
        re.compile(r"(?:daily reporting format).*?\b(\d{1,2}-\d{1,2}-\d{2})\b",re.I),
    ]

    alias_patterns=[]
    for canonical, aliases in {
        "Balaramapuram":["Balaramapuram"],
        "Pallichal":["Pallichal"],
        "Maranalloor":["Maranalloor"],
        "Malayinkeezhu":["Malayinkeezhu","Malayinkeezh","Malayinkeezh[u]?"],
        "Kalliyoor":["Kalliyoor"],
        "Vilappil":["Vilappil"],
        "Vilavoorkal":["Vilavoorkal","Vilavoor"] ,
        "Peringamala":["Peringamala"],
        "Karode":["Karode"],
        "Vellayani":["Vellayani"],
    }.items():
        for a in aliases:
            alias_patterns.append((canonical,re.compile(r"(?<![A-Za-z])"+re.escape(a)+r"(?![A-Za-z])",re.I)))

    def parse_report_date(text, filename):
        for pat in report_patters:
            m=pat.search(text)
            if m:
                raw=m.group(1)
                for dayfirst in [True,False]:
                    dt=pd.to_datetime(raw,errors="coerce",dayfirst=dayfirst)
                    if pd.notna(dt): return pd.Timestamp(dt).normalize()
        m=re.search(r"(\d{1,2})[._-](\d{1,2})[._-](20\d{2})",filename)
        if m:
            return pd.Timestamp(year=int(m.group(3)),month=int(m.group(2)),day=int(m.group(1)))
        return pd.NaT

    def locality_counts(text, district_total=None):
        """Extract locality-level dengue counts from the TVM locality field.

        IDSP uses two important patterns:

        1) ``TVM: 3 ; Kalliyoor, Pangappara, Andoorkonam``
           The 3 is the *district total*. The locality names identify where
           the reported cases occurred; because no locality-specific number is
           printed, each named locality is treated as one reported locality
           event.

        2) ``TVM: 5 ; Kalliyoor-3, Pangappara, Andoorkonam``
           Kalliyoor explicitly has 3 cases. The other named localities are
           treated as one case each unless they also carry an explicit count.

        Spacing around ``-``/``:`` and the number is intentionally flexible.
        The district total is retained separately for audit/validation and is
        never mistaken for a Kalliyoor count.
        """
        found=[]
        spans=[]
        for canonical,pat in alias_patterns:
            for m in pat.finditer(text or ""):
                spans.append((m.start(),m.end(),canonical,m.group(0)))
        spans.sort(key=lambda x:(x[0],-(x[1]-x[0])))
        used=[]
        for s,e,canonical,raw in spans:
            if any(s<ue and e>us for us,ue,_,_ in used):
                continue
            # Only a number directly attached to the locality is locality-
            # specific. Separators may be '-', en-dash, em-dash, ':', or
            # simply whitespace. A comma/semicolon does not qualify.
            tail=(text or "")[e:e+20]
            explicit=re.match(r"\s*(?:[-–—:]\s*)?(\d+)\b",tail)
            if explicit:
                count=float(explicit.group(1))
                source="explicit locality count"
            else:
                count=1.0
                source="implicit locality event"
            found.append((canonical,count,source))
            used.append((s,e,canonical,raw))
        return found

    total_files=max(1,len(files))
    for file_idx,(name,b) in enumerate(files):
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                total_pages=max(1,len(pdf.pages))
                if progress:
                    progress(file_idx/total_files, f"Reading IDSP report {file_idx+1}/{len(files)}: {name}")
                page_texts=[]
                for page_idx,page in enumerate(pdf.pages):
                    page_texts.append(page.extract_text() or "")
                    if progress:
                        # Health parsing is the slowest stage, so show page-level progress.
                        frac=(file_idx + (page_idx+1)/total_pages)/total_files
                        progress(frac, f"Reading IDSP report {file_idx+1}/{len(files)} — page {page_idx+1}/{total_pages}")
                full_text="\n".join(page_texts)
                report_date=parse_report_date(full_text,name)
                if pd.isna(report_date):
                    raise ValueError("Could not determine report date")
                reports_seen += 1
                report_dates.append(report_date)
                found_this_report=[]
                # The original IDSP layout places the disease/locality table on page 2.
                for page_idx,page in enumerate(pdf.pages):
                    if progress:
                        frac=(file_idx + 0.5 + 0.5*(page_idx/max(1,total_pages)))/total_files
                        progress(frac, f"Extracting dengue locality table — {name}, page {page_idx+1}/{total_pages}")
                    for table in (page.extract_tables() or []):
                        if not table: continue
                        dengue_row_index=None
                        for i,r in enumerate(table):
                            first=" ".join(str(v or "") for v in r[:3]).strip().lower()
                            if first == "dengue" or first.startswith("dengue "):
                                dengue_row_index=i; break
                        if dengue_row_index is None: continue
                        dengue_sections += 1
                        # Rows after Dengue contain district code, district case count and locality list.
                        for r in table[dengue_row_index:]:
                            cells=[str(v or "").strip() for v in r]
                            district_cell=cells[3] if len(cells)>3 else ""
                            if not district_cell: continue
                            dm=re.match(r"([A-Z]{2,3})\s*:?",district_cell.upper())
                            if not dm or dm.group(1)!="TVM": continue
                            loc_text=cells[5] if len(cells)>5 else ""
                            if not loc_text:
                                # Some PDF extraction variants put locality text in a later cell.
                                loc_text=" ".join(c for c in cells[5:] if c)
                            district_total_match=re.search(r"(?:TVM)\s*:?\s*(\d+(?:\.\d+)?)", district_cell, re.I)
                            district_total=float(district_total_match.group(1)) if district_total_match else np.nan
                            for loc,count,count_source in locality_counts(loc_text, district_total=district_total):
                                found_this_report.append((report_date,loc,float(count),district_total,count_source,loc_text.strip()))
                # Text fallback for the same TVM dengue row if table extraction is unavailable.
                if not found_this_report:
                    m=re.search(r"Dengue\s+\d+\s+\d+\s+TVM\s*:?\s*(\d+)\s+(.+?)(?=\s+KLM\s*:|\s+PTA\s*:|\s+Lepto\b)",full_text,re.I|re.S)
                    if m:
                        district_total=float(m.group(1)) if m.group(1) else np.nan
                        for loc,count,count_source in locality_counts(m.group(2), district_total=district_total):
                            found_this_report.append((report_date,loc,float(count),district_total,count_source,m.group(2).strip()))
                rows.extend(found_this_report)
        except Exception as e:
            # A bad/non-IDSP PDF should not silently become a zero-dengue day.
            raise ValueError(f"Could not parse {name}: {e}")

    if not rows:
        raise ValueError(
            f"Read {reports_seen} IDSP report(s), but found no Nemom dengue locality records. "
            "Check that the uploaded reports are the original Kerala IDSP daily reports and cover Nemom/TVM."
        )
    x=pd.DataFrame(rows,columns=["date","locality","dengue_cases","district_case_total","count_source","raw_locality_text"])
    if not x.empty:
        # A locality should normally occur once per report. If PDF extraction
        # creates duplicate rows, preserve the largest explicit/locality count
        # rather than summing duplicate mentions of the same locality.
        x["count_priority"]=x["count_source"].eq("explicit locality count").astype(int)
        x=x.sort_values(["date","locality","count_priority"],ascending=[True,True,False])
        x=x.groupby(["date","locality"],as_index=False).agg(
            dengue_cases=("dengue_cases","first"),
            district_case_total=("district_case_total","max"),
            count_source=("count_source","first"),
            raw_locality_text=("raw_locality_text","first")
        )
    # Preserve every successfully read report date, including days with no Nemom dengue locality.
    observed=pd.DataFrame({
        "date":pd.to_datetime(sorted(set(report_dates))),
        "locality":np.nan,
        "dengue_cases":np.nan,
        "district_case_total":np.nan,
        "count_source":np.nan,
        "raw_locality_text":np.nan
    })
    x=pd.concat([x,observed],ignore_index=True)
    x["source_type"]="IDSP daily report"
    return x

def extract_health(uploaded, progress=None):
    pdfs=[]; csvs=[]
    for f in uploaded:
        name=f.name.lower(); b=f.getvalue()
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                seen=set()
                for n in z.namelist():
                    if n in seen: continue
                    seen.add(n)
                    if n.lower().endswith(".pdf"):
                        pdfs.append((Path(n).name,z.read(n)))
                    elif n.lower().endswith((".csv",".txt")):
                        csvs.append((Path(n).name,z.read(n)))
        elif name.endswith(".pdf"):
            pdfs.append((f.name,b))
        elif name.endswith((".csv",".txt")):
            csvs.append((f.name,b))
    if csvs:
        try: return extract_health_from_csv(csvs)
        except Exception:
            if not pdfs: raise
    if pdfs: return extract_health_from_pdf(pdfs, progress=progress)
    raise ValueError("Upload original IDSP daily-report PDFs (individual PDFs or a ZIP containing them).")

def locality_from_text(text):
    """Identify a Nemom modelling locality from a sensor filename or device name."""
    s=str(text or "").lower()
    aliases=[
        ("Malayinkeezhu",["malayinkeezhu","malayinkeezh"]),
        ("Balaramapuram",["balaramapuram"]),
        ("Pallichal",["pallichal"]),
        ("Maranalloor",["maranalloor"]),
        ("Kalliyoor",["kalliyoor"]),
        ("Vilappil",["vilappil"]),
        ("Vilavoorkal",["vilavoorkal"]),
        ("Peringamala",["peringamala"]),
        ("Karode",["karode"]),
        ("Vellayani",["vellayani"]),
    ]
    for canonical,names in aliases:
        if any(n in s for n in names): return canonical
    return np.nan

def extract_iot(uploaded, progress=None):
    files=[]
    for f in uploaded:
        b=f.getvalue(); name=f.name
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                seen=set()
                for n in z.namelist():
                    if n in seen: continue
                    seen.add(n)
                    if n.lower().endswith(".csv"):
                        files.append((Path(n).name,z.read(n)))
        elif name.lower().endswith(".csv"):
            files.append((name,b))
    if not files: raise ValueError("Upload IoT CSV files or a ZIP containing CSV files.")
    rows=[]; diagnostics=[]
    for file_idx,(name,b) in enumerate(files):
        if progress:
            progress(file_idx/max(1,len(files)), f"Reading IoT file {file_idx+1}/{len(files)}: {name}")
        try: d=read_csv_bytes(b,name)
        except Exception as e: diagnostics.append((name,str(e))); continue
        dt=parse_date_col(d)
        if dt is None: diagnostics.append((name,"No date/time field")); continue
        sensor_col=find_col(d,["device_name","device","sensor","station","location","site","name"])
        if sensor_col is None:
            # filename often contains the sensor/site name
            sensor_values=pd.Series([Path(name).stem]*len(d),index=d.index)
        else:
            sensor_values=d[sensor_col].astype(str)
        temp=find_col(d,["temperature","temp"])
        hum=find_col(d,["humidity","relative_humidity","rh"])
        rain=find_col(d,["rainfall","rain"])
        wind=find_col(d,["wind_speed","windspeed","wind"])
        pressure=find_col(d,["pressure","barometric_pressure","atmospheric_pressure"])
        x=pd.DataFrame({"date":dt,"sensor":sensor_values,"source_file":Path(name).stem})
        for out,c in [("temperature",temp),("humidity",hum),("rainfall",rain),("wind_speed",wind),("pressure",pressure)]:
            x[out]=pd.to_numeric(d[c],errors="coerce") if c is not None else np.nan
        x=x[x.date.notna()]
        if x[["temperature","humidity","rainfall","wind_speed","pressure"]].notna().any().any():
            rows.append(x)
    if not rows: raise ValueError("No usable IoT measurements were detected.")
    x=pd.concat(rows,ignore_index=True)
    if progress:
        progress(1.0, f"IoT processing complete — {len(files)} files read")
    # Prefer the locality encoded in the original sensor filename/device name.
    # Fall back to the maintained sensor-site mapping for files that omit it.
    def map_sensor(row):
        direct=locality_from_text(row.get("source_file", ""))
        if pd.notna(direct): return direct
        direct=locality_from_text(row.get("sensor", ""))
        if pd.notna(direct): return direct
        for k,v in SENSOR_MAP_DEFAULT.items():
            if k.lower() in str(row.get("sensor", "")).lower(): return v
        return np.nan
    x["locality"]=x.apply(map_sensor,axis=1)
    return x, diagnostics

def extract_imd(uploaded, progress=None):
    grd=[]
    for f in uploaded:
        b=f.getvalue(); name=f.name
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                seen=set()
                for n in z.namelist():
                    if n in seen: continue
                    seen.add(n)
                    if n.lower().endswith(".grd"): grd.append((Path(n).name,z.read(n)))
        elif name.lower().endswith(".grd"): grd.append((name,b))
    if not grd: raise ValueError("Upload IMD .grd files or a ZIP containing them.")
    out=[]
    # The supplied IMD grids are daily float32 arrays, 129 x 135, 0.25-degree grid.
    # Use locality coordinates as a transparent modelling approximation.
    coords={"Balaramapuram":(8.432,77.032),"Pallichal":(8.475,77.006),"Maranalloor":(8.500,77.047),
            "Malayinkeezhu":(8.505,77.012),"Kalliyoor":(8.476,76.985),"Vilappil":(8.536,77.048),
            "Vilavoorkal":(8.524,77.039),"Peringamala":(8.575,77.048),"Karode":(8.385,77.065),"Vellayani":(8.429,76.999)}
    lat0,lon0,step,nlat,nlon=6.5,66.5,0.25,129,135
    total_grids=max(1,len(grd))
    for grid_idx,(name,b) in enumerate(grd):
        if progress:
            progress(grid_idx/total_grids, f"Reading IMD grid {grid_idx+1}/{len(grd)}: {name}")
        arr=np.frombuffer(b,dtype=np.float32)
        if arr.size % (nlat*nlon)!=0:
            continue
        days=arr.size//(nlat*nlon); arr=arr.reshape(days,nlat,nlon)
        year_match=re.search(r"(20\d{2})",name)
        year=int(year_match.group(1)) if year_match else None
        if year is None: continue
        dates=pd.date_range(f"{year}-01-01",periods=days,freq="D")
        for loc,(lat,lon) in coords.items():
            iy=int(round((lat-lat0)/step)); ix=int(round((lon-lon0)/step))
            iy=max(0,min(nlat-1,iy)); ix=max(0,min(nlon-1,ix))
            vals=arr[:,iy,ix].astype(float)
            out.append(pd.DataFrame({"date":dates,"locality":loc,"imd_rainfall":vals}))
        if progress:
            progress((grid_idx+1)/total_grids, f"Processed IMD grid {grid_idx+1}/{len(grd)}: {name}")
    if not out: raise ValueError("No supported IMD grids were found. Expected daily float32 129x135 .grd files.")
    x=pd.concat(out,ignore_index=True)
    return x.groupby(["date","locality"],as_index=False)["imd_rainfall"].mean()

def weekly_features(health,iot=None,imd=None, progress=None):
    if progress: progress(0.05, "Building weekly dengue records and IDSP coverage")
    raw_h=health.copy(); raw_h["date"]=pd.to_datetime(raw_h.date)
    observed_dates=pd.DatetimeIndex(sorted(raw_h["date"].dropna().dt.normalize().unique()))
    h=raw_h[raw_h.locality.isin(DENGUE_LOCALITIES)].copy(); h["week_start"]=dt_to_monday(h["date"])
    if h.empty:
        raise ValueError("No dengue records for the 10 configured Nemom modelling localities were found in the uploaded health reports.")
    hw=h.groupby(["locality","week_start"],as_index=False).agg(dengue_cases_current_week=("dengue_cases","sum"),
                                                               dengue_report_events_week=("dengue_cases","size"))
    # A day with no listed Nemom dengue locality is a zero-event day, but a missing
    # daily IDSP report must not silently be interpreted as zero. The upload coverage
    # is tracked separately below.
    daily_dates=pd.date_range(raw_h["date"].min().normalize(),raw_h["date"].max().normalize(),freq="D")
    coverage=pd.DataFrame({"date":daily_dates})
    coverage["idsp_report_observed"]=coverage["date"].isin(observed_dates)
    coverage["week_start"]=dt_to_monday(coverage["date"])
    week_cov=coverage.groupby("week_start",as_index=False)["idsp_report_observed"].agg(["mean","sum"])
    week_cov=week_cov.rename(columns={"mean":"idsp_daily_coverage","sum":"idsp_report_days"}).reset_index()
    full=pd.MultiIndex.from_product([DENGUE_LOCALITIES,pd.date_range(dt_to_monday(raw_h["date"]).min(),dt_to_monday(raw_h["date"]).max(),freq="7D")],
                                    names=["locality","week_start"]).to_frame(index=False)
    w=full.merge(hw,on=["locality","week_start"],how="left").merge(week_cov,on="week_start",how="left")
    # Only fully observed weeks are treated as valid zero/non-zero weekly outcomes.
    w["dengue_cases_current_week"]=w["dengue_cases_current_week"].fillna(0)
    w["dengue_report_events_week"]=w["dengue_report_events_week"].fillna(0)
    w.loc[w["idsp_daily_coverage"]<1,"dengue_cases_current_week"]=np.nan
    w.loc[w["idsp_daily_coverage"]<1,"dengue_report_events_week"]=np.nan
    w=w.sort_values(["locality","week_start"])
    for i in range(1,5): w[f"dengue_lag{i}"]=w.groupby("locality")["dengue_cases_current_week"].shift(i)
    w["dengue_cases_next_week"]=w.groupby("locality")["dengue_cases_current_week"].shift(-1)
    if progress: progress(0.45, "Adding weekly IMD rainfall")
    if imd is not None:
        x=imd.copy(); x["date"]=pd.to_datetime(x.date); x["week_start"]=dt_to_monday(x["date"])
        ir=x.groupby(["locality","week_start"],as_index=False)["imd_rainfall"].sum().rename(columns={"imd_rainfall":"rain"})
        w=w.merge(ir,on=["locality","week_start"],how="left")
    else: w["rain"]=np.nan
    if progress: progress(0.65, "Adding weekly IoT climate features")
    if iot is not None:
        x=iot.copy(); x["date"]=pd.to_datetime(x.date); x["week_start"]=dt_to_monday(x["date"])
        ag=x.groupby(["locality","week_start"],as_index=False).agg(
            iot_temperature_mean=("temperature","mean"),iot_humidity_mean=("humidity","mean"),
            iot_rainfall_weekly=("rainfall","sum"),iot_wind_speed_mean=("wind_speed","mean"),
            iot_pressure_mean=("pressure","mean"),iot_sensor_count=("sensor","nunique"),
            iot_sensor_days_mean=("date",lambda s:s.dt.date.nunique()))
        w=w.merge(ag,on=["locality","week_start"],how="left")
    else:
        for c in ["iot_temperature_mean","iot_humidity_mean","iot_rainfall_weekly","iot_wind_speed_mean","iot_pressure_mean","iot_sensor_count","iot_sensor_days_mean"]: w[c]=np.nan
    # Rainfall is optional in IoT data. Prefer IoT rainfall when available for a
    # locality-week; otherwise use the IMD gridded rainfall value. If neither
    # source is available, retain a missing value rather than assuming zero.
    iot_rain_available=w["iot_rainfall_weekly"].notna()
    imd_rain_available=w["rain"].notna()
    w["rainfall_source"]=np.select([iot_rain_available, imd_rain_available],["IoT sensor","IMD gridded"],default="Unavailable")
    w["rainfall_weekly"]=w["iot_rainfall_weekly"].where(iot_rain_available,w["rain"])
    w["iot_rainfall_weekly"]=w["rainfall_weekly"]
    for c in ["iot_temperature_mean","iot_humidity_mean","iot_rainfall_weekly","iot_wind_speed_mean"]:
        w[f"{c}_lag1"]=w.groupby("locality")[c].shift(1)
    for n in [14,21,28]:
        # Weekly equivalent rolling rainfall.
        w[f"rainfall_{n}d"]=w.groupby("locality")["rainfall_weekly"].transform(lambda s:s.rolling(max(1,n//7),min_periods=1).sum())
    w["year"]=w.week_start.dt.year; w["month"]=w.week_start.dt.month; w["week_of_year"]=w.week_start.dt.isocalendar().week.astype(int)
    w["week_sin"]=np.sin(2*np.pi*w["week_of_year"]/52.0); w["week_cos"]=np.cos(2*np.pi*w["week_of_year"]/52.0)
    w["target_week_fully_observed"]=w["dengue_cases_next_week"].notna()
    if progress: progress(1.0, "Feature engineering complete")
    return w

def dt_to_monday(s):
    s=pd.to_datetime(s)
    # Normalize time-of-day so daily health dates and timestamped IoT observations
    # join to the same Monday 00:00 weekly key.
    return s.dt.normalize()-pd.to_timedelta(s.dt.weekday,unit="D")

def week_coverage(df, week):
    """Return IDSP report coverage for a selected Monday-start week."""
    w=pd.Timestamp(week).normalize()
    if "idsp_daily_coverage" not in df.columns:
        return 1.0
    vals=df.loc[df[DATE]==w,"idsp_daily_coverage"].dropna()
    return float(vals.iloc[0]) if not vals.empty else 0.0


def make_model():
    return Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),
                     ("rf",RandomForestRegressor(n_estimators=500,max_depth=8,min_samples_leaf=2,max_features="sqrt",random_state=42,n_jobs=-1))])

def train(df,week):
    tr=df[(df[DATE]<week)&df[TARGET].notna()].copy()
    miss=[c for c in FEATURES if c not in tr.columns]
    if miss: raise ValueError("Missing model fields: "+", ".join(miss))
    if len(tr)<25: raise ValueError(f"Only {len(tr)} historical rows before the selected week; at least 25 are recommended.")
    model=make_model(); Xtr=tr[FEATURES].copy()
    for c in FEATURES:
        if Xtr[c].notna().sum()==0:
            Xtr[c]=0.0
    model.fit(Xtr,tr[TARGET].clip(lower=0))
    metrics={"MAE":np.nan,"RMSE":np.nan,"R2":np.nan,"n_train":len(tr),"n_test":0}
    weeks=sorted(tr[DATE].unique())
    if len(weeks)>=10:
        cut=max(1,int(len(weeks)*.8)); a=tr[tr[DATE].isin(weeks[:cut])]; b=tr[tr[DATE].isin(weeks[cut:])]
        if len(a)>=20 and len(b)>=3:
            em=make_model()
            Xa=a[FEATURES].copy(); Xb=b[FEATURES].copy()
            for c in FEATURES:
                if Xa[c].notna().sum()==0:
                    Xa[c]=0.0
            em.fit(Xa,a[TARGET].clip(lower=0)); p=np.maximum(em.predict(Xb),0)
            metrics={"MAE":mean_absolute_error(b[TARGET],p),"RMSE":mean_squared_error(b[TARGET],p)**.5,
                     "R2":r2_score(b[TARGET],p),"n_train":len(a),"n_test":len(b)}
    return model,metrics

def format_predicted_cases(value):
    """Show an integer case count first, with the unrounded model estimate in brackets."""
    if pd.isna(value):
        return "—"
    v=max(float(value),0.0)
    n=int(np.rint(v))
    label="case" if n==1 else "cases"
    return f"{n} {label} ({v:.2f} model estimate)"

def risk(vals):
    vals=np.asarray(vals,float)
    if len(vals)<4 or np.allclose(vals,vals[0]):
        return ["Moderate"]*len(vals)
    q25,q50,q75=np.nanpercentile(vals,[25,50,75])
    return ["Low" if x<=q25 else "Moderate" if x<=q50 else "High" if x<=q75 else "Very High" for x in vals]

@st.cache_data
def load_geo():
    return json.loads((BASE/"nemom_dashboard_geojson.json").read_text())

def centroids(geo):
    out={}
    def walk(x,pts):
        if isinstance(x,list) and len(x)>=2 and all(isinstance(v,(int,float)) for v in x[:2]): pts.append((x[1],x[0]))
        elif isinstance(x,list):
            for y in x: walk(y,pts)
    for f in geo["features"]:
        pts=[]; walk(f["geometry"]["coordinates"],pts)
        if pts: out[f["properties"]["LSGD"]]=(np.mean([p[0] for p in pts]),np.mean([p[1] for p in pts]))
    return out

def map_view(result,geo,selected):
    cs=centroids(geo)
    if selected in cs:
        selected_lat,selected_lon=cs[selected]
        m=folium.Map(location=[selected_lat,selected_lon],zoom_start=13,control_scale=True)
    else:
        m=folium.Map(location=[8.45,77.04],zoom_start=11,control_scale=True)

    title_html=f'''<div style="position: fixed; top: 12px; left: 50%; transform: translateX(-50%); z-index: 9999; background: rgba(255,255,255,.94); border: 2px solid #173f5f; border-radius: 8px; padding: 7px 14px; font-weight: 700; color: #173f5f; box-shadow: 0 1px 5px rgba(0,0,0,.2);">Selected locality: {selected}</div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    lookup=result.set_index(LOCALITY).to_dict("index")
    def style(f):
        n=f["properties"]["LSGD"]; d=lookup.get(n,{})
        is_selected=(n==selected)
        return {"color":"#111827" if is_selected else "#526777",
                "weight":5 if is_selected else 1.5,
                "fillColor":COLORS.get(d.get("risk"),"#b9c5ce"),
                "fillOpacity":.78 if is_selected else (.62 if d else .15)}
    folium.GeoJson(geo,name="Predicted risk",style_function=style,
                   tooltip=GeoJsonTooltip(fields=["LSGD","BLOCK","Area"],aliases=["Locality","Block","Area (sq km)"])).add_to(m)

    for _,r in result.iterrows():
        if r[LOCALITY] not in cs: continue
        lat,lon=cs[r[LOCALITY]]; p=max(float(r["predicted_next_week_cases"]),0)
        predicted_display=format_predicted_cases(p)
        html=f"<b>{r[LOCALITY]}</b><br>Current week: {r.get('dengue_cases_current_week',np.nan):.2f}<br>Predicted next week: {predicted_display}<br>Relative risk: <b>{r['risk']}</b>"
        folium.CircleMarker([lat,lon],radius=7+2*np.sqrt(p),color="#7b31b5",fill=True,fill_opacity=.85,
                            popup=folium.Popup(html,max_width=260)).add_to(m)

    if selected in cs:
        lat,lon=cs[selected]
        folium.Marker([lat,lon],tooltip=f"Selected: {selected}",
                      icon=folium.DivIcon(html=f'''<div style="font-size:12px;font-weight:700;color:#111827;text-shadow:0 1px 2px #fff;white-space:nowrap;">{selected}</div>''')).add_to(m)
    folium.LayerControl().add_to(m); return m
def data_status(label, ok, detail):
    st.write(("✅ " if ok else "⚠️ ")+f"**{label}** — {detail}")

st.markdown("""<div class="dashboard-hero">
<h1>Dengue Surveillance &amp; Forecast</h1>
<p>Nemom Block · Locality-level one-week forecasting using IDSP surveillance, environmental observations and IMD rainfall.</p>
</div>""", unsafe_allow_html=True)

# Session defaults
for k in ["health","iot","imd","processed","result","metrics"]:
    if k not in st.session_state: st.session_state[k]=None

with st.sidebar:
    st.header("Data & inputs")
    st.caption("Upload the original surveillance and environmental files. The application builds the weekly modelling dataset automatically.")
    health_files=st.file_uploader("Original IDSP daily reports",type=["pdf","zip","csv","txt"],accept_multiple_files=True,key="health_up")
    st.caption("Upload the original Kerala IDSP daily-report PDFs, individually or as one ZIP. Do not use the processed Health Summary PDF.")
    st.caption("IDSP locality values are derived from Nemom locality mentions in the daily Dengue section; explicit locality counts are used when printed in the report.")
    iot_files=st.file_uploader("IoT sensor data",type=["zip","csv"],accept_multiple_files=True,key="iot_up")
    imd_files=st.file_uploader("IMD rainfall (recommended fallback)",type=["zip","grd"],accept_multiple_files=True,key="imd_up")
    st.caption("IoT rainfall is optional. When sensor rainfall is unavailable, the application uses IMD gridded rainfall where available.")
    st.divider()
    process=st.button("⚙️ Process uploaded data",type="primary",use_container_width=True)
    st.caption("A prepared master CSV is no longer required. It is generated internally after processing.")

if process:
    progress_bar=st.progress(0, text="Starting data processing...")
    progress_status=st.empty()
    def show_progress(frac, message):
        frac=max(0.0,min(1.0,float(frac)))
        progress_bar.progress(int(frac*100), text=message)
        progress_status.caption(f"{int(frac*100)}% — {message}")
    try:
        # The three source stages occupy separate portions of the overall bar.
        show_progress(0.01, "Preparing uploaded files")
        def health_progress(f,msg): show_progress(0.05 + 0.30*f, msg)
        def iot_progress(f,msg): show_progress(0.35 + 0.25*f, msg)
        def imd_progress(f,msg): show_progress(0.60 + 0.25*f, msg)
        def feature_progress(f,msg): show_progress(0.85 + 0.15*f, msg)

        show_progress(0.03, "Extracting original IDSP daily reports")
        st.session_state.health=extract_health(health_files or [], progress=health_progress)
        show_progress(0.35, "IDSP health data extracted")

        st.session_state.iot,diag=extract_iot(iot_files or [], progress=iot_progress)
        show_progress(0.60, "IoT sensor data extracted")

        st.session_state.imd=extract_imd(imd_files or [], progress=imd_progress)
        show_progress(0.85, "IMD rainfall extracted")

        st.session_state.processed=weekly_features(
            st.session_state.health,st.session_state.iot,st.session_state.imd,
            progress=feature_progress
        )
        st.session_state.result=None
        show_progress(1.0, "Processing complete")
        st.success("Data processed successfully. Review the data-readiness summary below, then run the forecast.")
        if diag:
            st.warning("Some IoT files were skipped: "+ "; ".join(f"{n}: {m}" for n,m in diag[:8]))
    except Exception as e:
        progress_bar.progress(100, text="Processing stopped")
        progress_status.empty()
        st.error(str(e))

# Optional demo mode: lets the supplied prototype still be demonstrated without raw uploads.
if st.session_state.processed is None and not (health_files or iot_files or imd_files):
    demo=st.checkbox("Use bundled demonstration dataset (for testing only)")
    if demo:
        st.session_state.processed=pd.read_csv(BASE/"NEMOM_DENGUE_ML_MASTER.csv")
        st.session_state.processed[DATE]=pd.to_datetime(st.session_state.processed[DATE])
        st.info("Demo mode uses the bundled prepared dataset. For the real workflow, upload your source files above.")

st.markdown('<div class="section-heading">Data coverage & readiness</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Overview of the uploaded surveillance and environmental data available to the modelling workflow.</div>', unsafe_allow_html=True)
if st.session_state.processed is not None:
    p=st.session_state.processed
    h=st.session_state.health; io=st.session_state.iot; im=st.session_state.imd
    c=st.columns(4, gap="medium")
    with c[0]:
        with st.container(border=True):
            st.metric("Dengue records",f"{len(h):,}" if h is not None else "—")
            st.caption("IDSP locality-level records")
    with c[1]:
        with st.container(border=True):
            st.metric("IoT observations",f"{len(io):,}" if io is not None else "—")
            st.caption("Mapped sensor observations")
    with c[2]:
        with st.container(border=True):
            st.metric("IMD observations",f"{len(im):,}" if im is not None else "—")
            st.caption("Gridded rainfall records")
    with c[3]:
        with st.container(border=True):
            st.metric("Weekly locality rows",f"{len(p):,}")
            st.caption("Model-ready locality × week records")
    if h is not None: data_status("Dengue",True,f"{h.date.min():%d %b %Y} → {h.date.max():%d %b %Y}; {h.locality.nunique()} localities")
    if h is not None: st.caption("Health target is based on reported IDSP locality events extracted from the raw daily reports; it should not be interpreted as proof of disease absence/presence beyond the reporting system.")
    if "idsp_daily_coverage" in p.columns:
        cov=float(p["idsp_daily_coverage"].mean())*100
        st.caption(f"IDSP daily-report coverage across the uploaded period: {cov:.1f}% of calendar days represented.")
    if io is not None:
        data_status("IoT",True,f"{io.date.min():%d %b %Y} → {io.date.max():%d %b %Y}; {io.sensor.nunique()} sensors; {io.locality.notna().sum():,} observations mapped")
        if h is not None and (io.date.max() < h.date.min() or io.date.min() > h.date.max()):
            st.warning("IoT dates do not overlap the uploaded dengue-report period; IoT variables will not contribute to overlapping weeks.")
    if im is not None: data_status("IMD rainfall",True,f"{im.date.min():%d %b %Y} → {im.date.max():%d %b %Y}; {im.locality.nunique()} localities")
    unmapped=sorted(io.loc[io.locality.isna(),"sensor"].unique()) if io is not None else []
    if unmapped:
        st.warning("Unmapped IoT sensor names: "+", ".join(unmapped))
        st.caption("These sensors are excluded from locality-level modelling until mapped.")
    st.download_button("Download generated weekly dataset",p.to_csv(index=False).encode(),
                       file_name="NEMOM_weekly_features.csv",mime="text/csv")
else:
    st.info("Upload the original IDSP daily-report PDFs/ZIP, your IoT ZIP, and your IMD rainfall ZIP in the sidebar, then click **Process uploaded data**. A prepared master CSV is not required.")

# Forecast area
if st.session_state.processed is not None:
    df=st.session_state.processed.copy()
    df[DATE]=pd.to_datetime(df[DATE],errors="coerce")
    for c in FEATURES+[TARGET]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    all_weeks=sorted(df[DATE].dropna().unique())
    # Only fully observed current weeks are offered for forecasting. This prevents
    # a missing current-week dengue value from being silently median-imputed and
    # presented as a normal operational forecast.
    forecast_options=[w for w in all_weeks if week_coverage(df,pd.Timestamp(w))>=1.0]
    if all_weeks and not forecast_options:
        st.warning("No fully observed IDSP week is currently available for forecasting. Complete the daily reports for a full Monday–Sunday week, then rerun processing.")
    if forecast_options:
        st.markdown('<div class="section-heading">One-week dengue forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Choose a fully observed week as the forecast baseline, then estimate dengue cases for the following seven days.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="card-label">Forecast setup</div>', unsafe_allow_html=True)
            a,b,c=st.columns([1.55,1,1], gap="large")
            with a:
                week=st.selectbox("Forecast baseline week",forecast_options,index=len(forecast_options)-1,
                                  format_func=lambda x:pd.Timestamp(x).strftime("%d %b %Y"),
                                  help="The selected week provides the dengue history and environmental features used for the one-week-ahead forecast. Only weeks with all 7 IDSP reporting days represented are available for a standard forecast.")
            with b:
                st.markdown('<div class="card-label">Prediction horizon</div><div class="card-value">7 days</div><div class="card-note">Following Monday–Sunday week</div>', unsafe_allow_html=True)
            with c:
                st.markdown('<div class="card-label">Forecast model</div><div class="card-value">Random Forest</div><div class="card-note">Regression on locality-week features</div>', unsafe_allow_html=True)
            run=st.button("🚀 Generate one-week forecast",type="primary",use_container_width=True)
        if run or st.session_state.result is None:
            try:
                selected_week=pd.Timestamp(week)
                coverage=week_coverage(df,selected_week)
                if coverage < 1.0:
                    raise ValueError(
                        f"The selected forecast-from week ({selected_week:%d %b %Y}) has incomplete IDSP coverage "
                        f"({coverage*7:.0f}/7 reporting days represented). A forecast using an incomplete current week "
                        "is disabled because the current-week dengue feature would be imputed and could produce a misleading result. "
                        "Select a fully observed week instead."
                    )
                model,metrics=train(df,selected_week)
                result=df[df[DATE]==selected_week].copy()
                result["predicted_next_week_cases"]=np.maximum(model.predict(result[FEATURES]),0)
                result["risk"]=risk(result["predicted_next_week_cases"].values)
                target_week=selected_week+pd.Timedelta(days=7)
                observed=df.loc[df[DATE]==target_week,[LOCALITY,"dengue_cases_current_week"]].copy()
                observed=observed.rename(columns={"dengue_cases_current_week":"observed_next_week_cases"})
                result=result.merge(observed,on=LOCALITY,how="left")
                st.session_state.result=result; st.session_state.metrics=metrics
            except Exception as e:
                st.error(str(e)); st.stop()
        result=st.session_state.result.copy(); metrics=st.session_state.metrics
        result=result.sort_values("predicted_next_week_cases",ascending=False)
        cur=result["dengue_cases_current_week"].sum(min_count=1); pred=result["predicted_next_week_cases"].sum()
        hi=result["risk"].isin(["High","Very High"]).sum()
        observed_target=result["observed_next_week_cases"].sum(min_count=1) if "observed_next_week_cases" in result else np.nan
        target_start=pd.Timestamp(week)+pd.Timedelta(days=7)
        target_end=pd.Timestamp(week)+pd.Timedelta(days=13)

        st.markdown(
            f'<div class="section-heading">Modelled-area forecast summary</div>'
            f'<div class="section-subtitle">Aggregated across all modelled localities · Baseline: <b>{pd.Timestamp(week).strftime("%d %b %Y")}</b> · Target week: <b>{target_start.strftime("%d %b %Y")} – {target_end.strftime("%d %b %Y")}</b></div>',
            unsafe_allow_html=True
        )
        k=st.columns(5, gap="medium")
        kpi_specs=[
            ("Current reported cases","—" if pd.isna(cur) else f"{cur:.0f}",
             "Total dengue cases reported in the selected baseline week across all modelled localities. This is an aggregate for the modelled area, not the selected locality."),
            ("Predicted cases",format_predicted_cases(pred),
             "Total one-week-ahead Random Forest estimate across all modelled localities for the target week. The whole-number case count is shown first, with the unrounded model estimate in brackets."),
            ("High-risk localities",str(hi),
             "Number of modelled localities whose predicted values fall in the High or Very High relative-risk quartiles for the target week."),
            ("Validation MAE","—" if np.isnan(metrics["MAE"]) else f"{metrics['MAE']:.2f}",
             "Mean Absolute Error on the chronological holdout. Lower is better; it measures the average absolute prediction error."),
            ("Validation R²","—" if np.isnan(metrics["R2"]) else f"{metrics['R2']:.2f}",
             "R² on the chronological holdout. A negative value means performance was worse than the mean-value baseline on the held-out data.")
        ]
        for col,(label,value,help_text) in zip(k,kpi_specs):
            with col:
                with st.container(border=True):
                    st.metric(label,value,help=help_text)

        st.caption("Summary cards above are aggregated across the modelled localities. Select a locality below to view its individual forecast.")

        if pd.notna(observed_target):
            st.success(f"Observed target-week total: {observed_target:.0f} reported cases across the modelled localities. This is a later observed outcome used for comparison, not an input to the original forecast.")

        L,R=st.columns([1.7,1], gap="large")
        with L:
            st.markdown('<div class="section-heading">Forecast risk map</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Spatial view of the model-estimated dengue burden and relative-risk class for the target week.</div>', unsafe_allow_html=True)
            geo=load_geo(); selected_default=result.iloc[0][LOCALITY]
            selected=st.selectbox("Select locality",result[LOCALITY].tolist(),index=0,key="locality_select")
            st_folium(map_view(result,geo,selected),use_container_width=True,height=560,returned_objects=[])
            missing=sorted(set(result[LOCALITY])-set(centroids(geo)))
            if missing: st.caption("No polygon geometry in the current reference boundary file for: "+", ".join(missing))
        with R:
            st.markdown('<div class="section-heading">Locality details</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Individual locality view: forecast, relative risk and environmental indicators for the selected locality.</div>', unsafe_allow_html=True)
            row=result[result[LOCALITY]==selected].iloc[0]
            with st.container(border=True):
                pred_value=float(row["predicted_next_week_cases"])
                st.metric(
                    "Predicted target-week cases",
                    format_predicted_cases(pred_value),
                    help="This is a one-week-ahead Random Forest regression estimate. Because regression outputs are continuous, the underlying estimate can be decimal. The dashboard shows the nearest whole number of cases first and keeps the original unrounded estimate in brackets."
                )
                st.metric(
                    "Relative risk",
                    row["risk"],
                    help="Relative risk is assigned from quartiles of the unrounded predicted case counts across the modelled localities. Very High means the locality is in the highest prediction quartile for this forecast week; it does not mean a clinical probability or a guaranteed high number of cases."
                )
            st.caption("Highest prediction quartile among modelled localities." if row["risk"]=="Very High" else "Risk is relative to the modelled localities for the selected forecast week.")
            if pd.notna(row.get("observed_next_week_cases",np.nan)):
                observed_value=float(row["observed_next_week_cases"])
                observed_label=f"{int(np.rint(observed_value))} case" if int(np.rint(observed_value))==1 else f"{int(np.rint(observed_value))} cases"
                st.metric(
                    "Observed target-week cases",
                    observed_label,
                    help="Observed dengue cases reported for the forecast target week after that week has occurred. This is used for forecast evaluation and is not fed back into the original forecast."
                )
                err=observed_value-pred_value
                st.caption(f"Forecast error (observed − predicted): {err:+.2f} cases")
            vals={"Current-week cases":row.get("dengue_cases_current_week",np.nan),
                  "Previous week":row.get("dengue_lag1",np.nan),"Rainfall (week)":row.get("iot_rainfall_weekly",np.nan),
                  "Temperature":row.get("iot_temperature_mean",np.nan),"Humidity":row.get("iot_humidity_mean",np.nan),
                  "Wind speed":row.get("iot_wind_speed_mean",np.nan),"Rainfall source":row.get("rainfall_source","Unavailable")}
            st.dataframe(pd.DataFrame(vals.items(),columns=["Indicator","Value"]),hide_index=True,use_container_width=True)
            st.caption("Relative risk is a dashboard classification of the unrounded model-estimated case counts using quartiles across the forecast localities; it is not a clinical probability and does not mean the model expects a fractional patient.")

        with st.expander("How to read the forecast and relative-risk classes", expanded=False):
            st.markdown(
                "**Forecast value:** the number before the brackets is the nearest whole-number case count for practical reading; the value in brackets is the original continuous Random Forest estimate. "
                "A value such as **0 cases (0.34 model estimate)** means the model estimates 0.34 expected cases — it does not represent a fraction of a patient.\n\n"
                "**Relative risk:** Low, Moderate, High and Very High are quartile-based dashboard categories calculated from the unrounded predictions across the modelled localities. "
                "**Very High therefore means the prediction is in the highest prediction quartile, not that a specific number of cases is guaranteed.**"
            )

        st.markdown('<div class="section-heading">Locality forecast results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Side-by-side forecast output for all modelled localities. Predictions show the practical whole-number count followed by the unrounded model estimate.</div>', unsafe_allow_html=True)
        cols=[c for c in [LOCALITY,"dengue_cases_current_week","predicted_next_week_cases","observed_next_week_cases","risk",
                          "iot_rainfall_weekly","rainfall_source","iot_temperature_mean","iot_humidity_mean"] if c in result]
        display_result=result[cols].copy()
        if "dengue_cases_current_week" in display_result:
            display_result["dengue_cases_current_week"]=display_result["dengue_cases_current_week"].apply(
                lambda x: "—" if pd.isna(x) else f"{int(np.rint(float(x)))}"
            )
        if "predicted_next_week_cases" in display_result:
            display_result["predicted_next_week_cases"]=display_result["predicted_next_week_cases"].apply(format_predicted_cases)
        if "observed_next_week_cases" in display_result:
            display_result["observed_next_week_cases"]=display_result["observed_next_week_cases"].apply(
                lambda x: "—" if pd.isna(x) else f"{int(np.rint(float(x)))}"
            )
        display_result=display_result.rename(columns={
            LOCALITY:"Locality",
            "dengue_cases_current_week":"Baseline week / reported cases",
            "predicted_next_week_cases":"Forecast target / estimated cases",
            "observed_next_week_cases":"Target week / observed cases",
            "risk":"Relative risk",
            "iot_rainfall_weekly":"Weekly rainfall (mm)",
            "rainfall_source":"Rainfall source",
            "iot_temperature_mean":"Mean temperature (°C)",
            "iot_humidity_mean":"Mean humidity (%)"
        })
        st.dataframe(display_result,hide_index=True,use_container_width=True)

        hist=df[df[LOCALITY]==selected].sort_values(DATE).tail(16).copy()
        if not hist.empty:
            st.markdown('<div class="section-heading">Recent surveillance & environmental trends</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Review recent reported dengue counts alongside rainfall or humidity for the selected locality.</div>', unsafe_allow_html=True)
            comparison=st.radio("Trend comparison",["Dengue cases only","Dengue vs rainfall","Dengue vs humidity"],horizontal=True,key="trend_comparison")
            if comparison=="Dengue cases only":
                fig=px.line(hist,x=DATE,y="dengue_cases_current_week",markers=True,
                            title=f"{selected} — observed weekly dengue cases")
                fig.update_yaxes(title="Reported dengue cases")
            else:
                env_col="rainfall_weekly" if comparison=="Dengue vs rainfall" else "iot_humidity_mean"
                env_label="Rainfall (mm/week)" if comparison=="Dengue vs rainfall" else "Humidity (%)"
                if env_col not in hist.columns: hist[env_col]=np.nan
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                fig=make_subplots(specs=[[{"secondary_y":True}]])
                fig.add_trace(go.Scatter(x=hist[DATE],y=hist["dengue_cases_current_week"],mode="lines+markers",name="Dengue cases"),secondary_y=False)
                fig.add_trace(go.Scatter(x=hist[DATE],y=hist[env_col],mode="lines+markers",name=env_label),secondary_y=True)
                fig.update_yaxes(title_text="Reported dengue cases",secondary_y=False)
                fig.update_yaxes(title_text=env_label,secondary_y=True)
                fig.update_layout(title=f"{selected} — dengue cases vs {env_label.lower()}")
            fig.update_layout(height=340,margin=dict(l=10,r=10,t=55,b=10),legend=dict(orientation="h",y=1.08))
            st.plotly_chart(fig,use_container_width=True)

        st.markdown('<div class="section-heading">Model performance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Chronological holdout evaluation of the Random Forest. Use these metrics to judge how well the model has performed historically before interpreting the forecast map.</div>',
            unsafe_allow_html=True
        )
        with st.container(border=True):
            v1,v2,v3=st.columns(3,gap="large")
            with v1:
                st.metric("Validation MAE","—" if np.isnan(metrics["MAE"]) else f"{metrics['MAE']:.2f}",
                          help="Mean Absolute Error on the chronological validation set. Lower is better.")
            with v2:
                st.metric("Validation RMSE","—" if np.isnan(metrics["RMSE"]) else f"{metrics['RMSE']:.2f}",
                          help="Root Mean Squared Error on the chronological validation set. Larger errors are penalised more strongly.")
            with v3:
                st.metric("Validation R²","—" if np.isnan(metrics["R2"]) else f"{metrics['R2']:.2f}",
                          help="R² on the chronological validation set. A negative value means the model performed worse than the mean-value baseline on the held-out data.")
            st.caption(f"Chronological holdout: {metrics['n_test']} test rows after {metrics['n_train']} training rows. Metrics change with the uploaded dataset.")
        export_result=result.copy()
        export_result["predicted_next_week_cases"]=export_result["predicted_next_week_cases"].round(4)
        st.download_button("Download forecast CSV",export_result.to_csv(index=False).encode(),
                           file_name="NEMOM_Dengue_1Week_Forecast.csv",mime="text/csv")
else:
    st.markdown('<div class="section-heading">Getting started</div>', unsafe_allow_html=True)
    st.write("**Upload → Validate → Process → Forecast → Review → Export**")
    st.caption("The bundled master CSV is retained only as an optional demonstration dataset; it is not required for the real workflow.")
