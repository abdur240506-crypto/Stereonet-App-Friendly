# app.py — Stereonet Geologi Struktur
# jalanin di terminal: streamlit run app.py
# install dulu: pip install streamlit mplstereonet matplotlib numpy obspy

# library utama streamlit buat bikin tampilannya jadi web
import streamlit as st

# numpy buat hitung-hitung vektor, trigonometri, array — wajib ada
import numpy as np

# matplotlib buat gambar semua plotnya (stereonet, rose, beachball, dll)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Counter dari collections buat ngitung frekuensi tiap tipe Riedel
from collections import Counter

# mplstereonet nambahin proyeksi stereonet ke matplotlib
# tanpa ini ga bisa bikin ax dengan projection='stereonet'
import mplstereonet

# obspy itu library seismologi, kita cuma butuh satu fungsi: beach() buat beachball
# dibungkus try/except karena obspy agak berat, kalau ga keinstall app tetap jalan
try:
    from obspy.imaging.beachball import beach
    OBSPY_OK = True
except ImportError:
    OBSPY_OK = False


# ── konfigurasi halaman ──────────────────────────────────────
# ini harus paling atas sebelum st yang lain, kalau ngga error
# layout wide biar kolom input + plot bisa side by side
st.set_page_config(
    page_title="Stereonet Geologi Struktur",
    page_icon="🪨",
    layout="wide",
)

# CSS custom buat polish tampilannya
# IBM Plex Sans buat teks biasa, IBM Plex Mono buat angka & judul — keliatan lebih technical
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* background krem muda biar ga terlalu putih, lebih enak dilihat */
.main { background-color: #F7F5F0; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* label input dikecilkan biar ga makan space */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-size: 13px;
    color: #555;
}

/* tombol hitam solid, hover jadi abu gelap */
.stButton button {
    background: #1a1a1a;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    padding: 6px 16px;
    transition: background 0.15s;
}
.stButton button:hover { background: #333; }

/* kotak metrik buat nampilin hasil angka */
.metric-box {
    background: #fff;
    border: 1px solid #e0ddd6;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
.metric-value {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
    font-family: 'IBM Plex Mono', monospace;
}

/* kotak info/interpretasi dengan garis biru di kiri */
.info-box {
    background: #EEF4FB;
    border-left: 3px solid #185FA5;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #1a3a5c;
    margin-top: 10px;
    line-height: 1.7;
}

/* kotak info hijau khusus buat kekar */
.info-box-green {
    background: #EDFAF3;
    border-left: 3px solid #1D9E75;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    color: #0f3d26;
    margin-top: 10px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# palet warna buat set data — 8 warna biar cukup walau input banyak
# warnanya dipilih yang kontras satu sama lain di background putih
COLORS = [
    '#185FA5',  # biru
    '#D85A30',  # oranye
    '#1D9E75',  # hijau teal
    '#D4537E',  # pink
    '#BA7517',  # amber
    '#7F77DD',  # ungu
    '#0F6E56',  # hijau tua
    '#993C1D',  # coklat
]


# ════════════════════════════════════════════════════════════
# FUNGSI MATEMATIKA & GEOMETRI
# ════════════════════════════════════════════════════════════

def mean_angle(angles):
    # rata-rata sudut yang bener, bukan pakai sum/n biasa
    # masalahnya: rata-rata 10° dan 350° harusnya 0°, bukan 180°
    # solusi: uraikan ke sin & cos dulu, baru arctan2
    r = np.radians(angles)
    return float((np.degrees(np.arctan2(
        np.mean(np.sin(r)), np.mean(np.cos(r))
    )) + 360) % 360)


def angular_diff(a, b):
    # selisih sudut terkecil antara dua arah (hasil selalu 0–180°)
    # pakai modulo 360 dulu biar ga negative, trus ambil yang terkecil
    d = abs(a - b) % 360
    return min(d, 360 - d)


def plane_to_vec(strike, dip):
    # konversi bidang geologi (strike/dip) ke vektor 3D
    # vektor ini adalah pole = garis tegak lurus bidang
    # rumusnya dari geometri bola: sin(dip)*sin(strike) dst
    s = np.radians(strike)
    d = np.radians(dip)
    v = np.array([
        np.sin(d) * np.sin(s),   # arah timur (X)
        np.sin(d) * np.cos(s),   # arah utara (Y)
        np.cos(d)                 # arah atas (Z)
    ])
    if v[2] < 0:
        v = -v   # pastiin selalu di lower hemisphere
    return v / np.linalg.norm(v)   # normalisasi jadi unit vektor


def normalize(v):
    # jadiin vektor jadi panjang 1 (unit vektor)
    # ada pengecekan kecil biar ga dibagi 0
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def vec_to_trend_plunge(v):
    # balik lagi dari vektor 3D ke trend/plunge yang bisa dibaca manusia
    # trend = arah horizontal dari utara (0–360°)
    # plunge = sudut turun dari horizontal (0–90°)
    x, y, z = v
    trend  = float((np.degrees(np.arctan2(x, y)) + 360) % 360)
    plunge = float(np.degrees(np.arcsin(np.clip(abs(z), -1, 1))))
    if z < 0:
        trend = (trend + 180) % 360
    return trend, plunge


def classify_fold_fleuty(plunge, interlimb_angle=None):
    # klasifikasi lipatan Fleuty (1964) — dua parameter:
    # 1. plunge sumbu lipatan → posisi sumbu di ruang
    # 2. interlimb angle (sudut antar dua limb) → tingkat keketatan lipatan
    #
    # Klasifikasi orientasi sumbu (dari plunge):
    # Recumbent       : plunge 0–10°  (hampir rebah)
    # Subhorizontal   : 0–10° (beberapa referensi samain dengan recumbent)
    # Gently plunging : 10–30°
    # Moderately      : 30–60°
    # Steeply         : 60–80°
    # Subvertical     : 80–90°
    if plunge < 10:        plunge_class = "Recumbent / subhorizontal"
    elif plunge < 30:      plunge_class = "Gently plunging"
    elif plunge < 60:      plunge_class = "Moderately plunging"
    elif plunge < 80:      plunge_class = "Steeply plunging"
    else:                  plunge_class = "Subvertical / upright"

    # Klasifikasi tightness dari interlimb angle (sudut antar limb):
    # Gentle    : >120°  (lipatan landai, hampir datar)
    # Open      : 70–120°
    # Close     : 30–70°
    # Tight     : 5–30°  (sangat terlipat)
    # Isoclinal : <5°    (kedua limb hampir paralel)
    if interlimb_angle is None:
        tight_class = "interlimb angle tidak diisi"
    elif interlimb_angle > 120:  tight_class = "Gentle fold (>120°)"
    elif interlimb_angle > 70:   tight_class = "Open fold (70°–120°)"
    elif interlimb_angle > 30:   tight_class = "Close fold (30°–70°)"
    elif interlimb_angle > 5:    tight_class = "Tight fold (5°–30°)"
    else:                        tight_class = "Isoclinal fold (<5°)"

    return plunge_class, tight_class


def calc_interlimb_angle(fold_left, fold_right):
    # hitung interlimb angle = sudut antara mean pole kiri dan kanan
    # kenapa dari pole? karena sudut antar pole = sudut antar bidang
    # (suplemen dari sudut antar normal = sudut antar limb itu sendiri)
    def mv(sets):
        vecs = [plane_to_vec(s['strike'], s['dip']) for s in sets]
        return normalize(np.mean(vecs, axis=0))
    mL = mv(fold_left)
    mR = mv(fold_right)
    # dot product dua unit vektor = cos(sudut antar keduanya)
    cos_a = np.clip(np.dot(mL, mR), -1, 1)
    angle_between_poles = np.degrees(np.arccos(cos_a))
    # interlimb angle = 180° - sudut antar pole (karena pole tegak lurus bidang)
    interlimb = 180.0 - angle_between_poles
    return float(interlimb)


def classify_fault_by_rake(rake):
    # klasifikasi sesar dari rake — ini yang paling akurat
    # rake = sudut arah slip di permukaan bidang sesar (-180 sampai +180°)
    #
    # Referensi: Aki & Richards (2002), Twiss & Moores (2007)
    # rake  0° atau ±180° = pure strike-slip (gerak horizontal)
    # rake +90°            = pure reverse / thrust (naik)
    # rake -90°            = pure normal fault (turun)
    # zona ±30° dari 0/180 = dominan strike-slip
    # zona ±30° dari 90   = dominan dip-slip
    rake_abs = abs(rake)
    if rake_abs <= 30 or rake_abs >= 150:
        return "Strike-slip"
    elif 60 <= rake_abs <= 120 and rake < 0:
        return "Normal fault"
    elif 60 <= rake_abs <= 120 and rake > 0:
        return "Reverse / thrust"
    else:
        return "Oblique slip"


def classify_fault_by_dip(dips):
    # fallback klasifikasi kalau pitch semua = 0 (belum diisi manual)
    # pakai dip rata-rata sebagai proxy — kurang akurat tapi cukup untuk estimasi
    # catatan: sesar strike-slip bisa dip-nya curam (~90°), makanya ini bisa salah
    # → selalu lebih baik isi pitch/rake secara manual untuk hasil yang bener
    arr = np.array(dips)
    n   = len(arr)
    if np.sum(arr > 60) / n > 0.5:                    return "Normal fault dominant*"
    if np.sum(arr < 30) / n > 0.5:                    return "Strike-slip dominant*"
    if np.sum((arr >= 30) & (arr <= 60)) / n > 0.5:   return "Reverse / thrust dominant*"
    return "Oblique slip / mixed*"
    # tanda * berarti estimasi dari dip, bukan dari rake


def estimate_rake(mean_dip, sense):
    # estimasi rake otomatis kalau pengguna ga ngisi pitch
    # ini cuma fallback kasar — hasilnya bisa salah seperti kasus di atas
    # (dip 85° bisa jadi strike-slip kalau rake-nya kecil)
    # → isi pitch manual kalau mau klasifikasi yang bener
    if mean_dip > 60:   return -90
    elif mean_dip < 30: return 0
    else:               return 30 if 'Dextral' in sense else -30


def classify_joint(dip_angle):
    # klasifikasi kekar berdasarkan sudut dip-nya
    # sistemnya sederhana: vertikal, subvertikal, miring, subhorizontal
    if dip_angle >= 80:          return "Kekar vertikal (≥ 80°)"
    elif dip_angle >= 60:        return "Kekar subvertikal (60°–79°)"
    elif dip_angle >= 30:        return "Kekar miring / diagonal (30°–59°)"
    else:                        return "Kekar subhorizontal / lembaran (< 30°)"


def classify_joint_system(strikes):
    # cek apakah kekar-kekarnya punya orientasi yang sistematis (>1 set arah)
    # caranya: hitung selisih tiap pasang, kalau ada yang ~90° berarti ada 2 set
    # ini cara sederhana, untuk data lebih banyak idealnya pakai clustering
    if len(strikes) < 2:
        return "Data kurang untuk analisis sistem"
    diffs = []
    for i in range(len(strikes)):
        for j in range(i + 1, len(strikes)):
            diffs.append(angular_diff(strikes[i], strikes[j]))
    min_diff = min(diffs)
    if min_diff < 20:
        return "Satu set dominan (kekar paralel)"
    elif 70 <= min_diff <= 110:
        return "Dua set saling tegak lurus (sistem kekar ortogonal)"
    else:
        return "Sistem kekar conjugate / tidak beraturan"


# ════════════════════════════════════════════════════════════
# FUNGSI PLOT
# ════════════════════════════════════════════════════════════

def plot_fault_stereonet(fault_sets):
    # stereonet utama buat data sesar
    # tiap set di-plot great circle (bidang) + pole (titik tegak lurus)
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5),
        subplot_kw=dict(projection='stereonet')  # ini yang bikin jadi lingkaran stereonet
    )
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    legend_handles = []
    for i, fs in enumerate(fault_sets):
        col = COLORS[i % len(COLORS)]   # cycling warna biar ga error kalau data > 8

        # ax.plane() gambar great circle = representasi bidang penuh di stereonet
        ax.plane(fs['strike'], fs['dip'], color=col, linewidth=1.5, alpha=0.7)

        # ax.pole() gambar titik pole = lebih enak dibaca kalau data banyak
        ax.pole(fs['strike'], fs['dip'], color=col, marker='o', markersize=7)

        lbl = "Set %d  %d°/%d°" % (i + 1, fs['strike'], fs['dip'])
        if fs.get('pitch', 0):
            lbl += "  pitch %d°" % fs['pitch']   # tambahin pitch ke label kalau diisi
        legend_handles.append(Line2D([0], [0], color=col, linewidth=2, label=lbl))

    ax.grid(True, alpha=0.3)

    # set_azimuth_ticks & ticklabels — ini kita override label default mplstereonet
    # karena label defaultnya sering ketimpa frame atau posisinya aneh
    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.set_azimuth_ticklabels(['N', 'E', 'S', 'W'], fontsize=10, fontweight='bold')

    ax.set_title("Stereonet Sesar\n(Great Circle & Pole)", fontsize=11, pad=20)

    # legend ditaruh di bawah dengan bbox_to_anchor supaya ga nutupin stereonet
    # ncol=2 biar ga terlalu panjang ke bawah kalau banyak set
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5,
              framealpha=0.9, edgecolor='#ccc',
              bbox_to_anchor=(-0.05, -0.20), ncol=2)
    fig.subplots_adjust(bottom=0.22)
    return fig


def plot_rose(strikes):
    # rose diagram = histogram di polar axes, buat lihat orientasi dominan strike
    fig = plt.figure(figsize=(4.5, 4.5))
    ax  = fig.add_subplot(111, polar=True)   # polar=True yang bikin jadi melingkar
    fig.patch.set_facecolor('#FAFAF8')
    ax.set_facecolor('#FAFAF8')

    # data strike didobel dengan +180° karena satu bidang punya dua arah berlawanan
    # kalau ga didobel rose-nya asimetris padahal secara geologi harusnya simetris
    all_s = np.concatenate([strikes, (strikes + 180) % 360])
    bins  = np.linspace(0, 2 * np.pi, 37)   # 36 bins × 10° per bin
    n, _  = np.histogram(np.radians(all_s), bins=bins)

    # utara di atas, searah jarum jam = konvensi standar kompas geologi
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.bar(bins[:-1], n, width=2 * np.pi / 36,
           bottom=0, color='#185FA5', edgecolor='white', alpha=0.8, linewidth=0.5)
    ax.set_yticks([])   # hapus label jari-jari, kurang informatif di rose
    ax.set_title("Rose Diagram (Strike)", fontsize=11, pad=14)
    plt.tight_layout()
    return fig


def plot_stress_stereonet(sigma1, sigma2, sigma3):
    # stereonet buat nampilin posisi sumbu tegasan utama (σ1, σ2, σ3)
    # σ1 = kompresi maksimum, σ3 = ekstensi, σ2 = intermediate
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5),
        subplot_kw=dict(projection='stereonet')
    )
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    for trend, col, lbl, ls, mk in [
        (sigma1, '#1D9E75', 'σ1 (%.0f°)' % sigma1, '-',  '^'),
        (sigma2, '#BA7517', 'σ2 (%.0f°)' % sigma2, '--', 's'),
        (sigma3, '#D85A30', 'σ3 (%.0f°)' % sigma3, ':',  'v'),
    ]:
        # strike bidang tegasan = trend - 90° karena tegak lurus sumbu-nya
        ax.plane((trend - 90) % 360, 90, color=col, linewidth=2, linestyle=ls)
        # plunge=0 karena asumsi sumbu horizontal (penyederhanaan)
        ax.line(0, trend, marker=mk, color=col, markersize=11, label=lbl)

    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.set_azimuth_ticklabels(['N', 'E', 'S', 'W'], fontsize=10, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9,
              edgecolor='#ccc', bbox_to_anchor=(-0.05, -0.12))
    ax.grid(True, alpha=0.3)
    ax.set_title("Stress Axes (σ1, σ2, σ3)", fontsize=11, pad=20)
    fig.subplots_adjust(bottom=0.14)
    return fig


def plot_beachball(strike, dip, rake):
    # beachball = focal mechanism, nampilin pola slip sesar
    # kuadran gelap = compressional (daerah tekan, P-axis)
    # kuadran terang = dilatational (daerah tarik, T-axis)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')
    ax.set_aspect('equal')
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis('off')

    if OBSPY_OK:
        try:
            # beach() dari obspy adalah cara paling akurat gambar beachball
            # input: [strike, dip, rake] dari bidang sesar utama
            b = beach(
                [strike, dip, rake],
                xy=(0, 0), width=1.9,
                linewidth=1, facecolor='#185FA5', alpha=0.85
            )
            ax.add_collection(b)
        except Exception as e:
            ax.text(0, 0, "Error:\n" + str(e), ha='center', va='center', fontsize=9)
    else:
        # kalau obspy ga ada, gambar manual sederhana
        # ini cuma approximasi visual, bukan proyeksi yang akurat
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.fill(np.cos(theta) * 0.9, np.sin(theta) * 0.9, color='#185FA5', alpha=0.85)
        ax.plot(np.cos(theta) * 0.9, np.sin(theta) * 0.9, 'k-', linewidth=1.5)
        p_dir = np.radians(strike + (45 if rake >= 0 else -45))
        t_dir = np.radians(strike + (-45 if rake >= 0 else 45))
        ax.text(0.5 * np.sin(p_dir), 0.5 * np.cos(p_dir), 'P',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(0.5 * np.sin(t_dir), 0.5 * np.cos(t_dir), 'T',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    ax.set_title(
        "Beachball  Strike %d° / Dip %d° / Rake %d°" % (strike, dip, rake),
        fontsize=10, pad=10
    )
    handles = [
        mpatches.Patch(color='#185FA5', label='Compressional quadrant (P-axis)'),
        mpatches.Patch(color='white', label='Dilatational quadrant (T-axis)',
                       edgecolor='#185FA5'),
    ]
    ax.legend(handles=handles, loc='lower center', fontsize=8,
              framealpha=0.92, edgecolor='#ccc', bbox_to_anchor=(0.5, -0.10))
    fig.subplots_adjust(bottom=0.16)
    return fig


def plot_fold_stereonet(fold_left, fold_right):
    # stereonet lipatan: plot kedua limb + hitung sumbu lipatan
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5),
        subplot_kw=dict(projection='stereonet')
    )
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    # plot semua data limb kiri (biru solid) dan kanan (oranye putus-putus)
    for fs in fold_left:
        ax.plane(fs['strike'], fs['dip'], color='#185FA5', linewidth=1.5, alpha=0.6)
        ax.pole(fs['strike'],  fs['dip'], color='#185FA5', marker='o', markersize=6)

    for fs in fold_right:
        ax.plane(fs['strike'], fs['dip'], color='#D85A30',
                 linewidth=1.5, linestyle='--', alpha=0.6)
        ax.pole(fs['strike'],  fs['dip'], color='#D85A30', marker='o', markersize=6)

    def mean_vec(sets):
        # rata-rata vektor = hitung mean di ruang 3D Cartesian
        # lebih stabil dari rata-rata sudut langsung,
        # terutama kalau data nyebrang 0°/360°
        vecs = [plane_to_vec(s['strike'], s['dip']) for s in sets]
        return normalize(np.mean(vecs, axis=0))

    mL = mean_vec(fold_left)    # mean pole limb kiri
    mR = mean_vec(fold_right)   # mean pole limb kanan

    # fold axis = cross product dua mean pole
    # kenapa cross product? karena cross product dari dua normal bidang
    # = vektor yang sejajar dengan garis perpotongan kedua bidang
    # = itu definisi sumbu lipatan secara geometri 3D
    fv = normalize(np.cross(mL, mR))
    if fv[2] < 0:
        fv = -fv   # flip ke lower hemisphere biar konsisten
    if np.isnan(fv[0]):
        # kalau dua limb paralel, cross product = 0 → fallback ke rata-rata
        fv = normalize((mL + mR) / 2)

    tL, pL = vec_to_trend_plunge(mL)
    tR, pR = vec_to_trend_plunge(mR)
    tA, pA = vec_to_trend_plunge(fv)

    # segitiga (^) buat mean pole — beda dari lingkaran data individual
    ax.line(pL, tL, marker='^', color='#185FA5', markersize=13, zorder=5)
    ax.line(pR, tR, marker='^', color='#D85A30', markersize=13, zorder=5)

    # bintang hijau (*) buat fold axis — ukuran gede biar gampang ketemu
    ax.line(pA, tA, marker='*', color='#1D9E75', markersize=20, zorder=6)

    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.set_azimuth_ticklabels(['N', 'E', 'S', 'W'], fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_title("Stereonet Lipatan\n(Great Circle & Fold Axis)", fontsize=11, pad=20)

    # legend manual yang eksplisit biar ga ada simbol misterius
    handles = [
        Line2D([0],[0], color='#185FA5', linewidth=2,
               label='Left limb (great circle)'),
        Line2D([0],[0], color='#185FA5', marker='o', linestyle='None',
               markersize=7, label='Left limb (pole)'),
        Line2D([0],[0], color='#185FA5', marker='^', linestyle='None',
               markersize=9, label='Mean pole – left'),
        Line2D([0],[0], color='#D85A30', linewidth=2, linestyle='--',
               label='Right limb (great circle)'),
        Line2D([0],[0], color='#D85A30', marker='o', linestyle='None',
               markersize=7, label='Right limb (pole)'),
        Line2D([0],[0], color='#D85A30', marker='^', linestyle='None',
               markersize=9, label='Mean pole – right'),
        Line2D([0],[0], color='#1D9E75', marker='*', linestyle='None',
               markersize=14, label='Fold axis (%.0f°/%.0f°)' % (tA, pA)),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=7.5,
              framealpha=0.92, edgecolor='#ccc',
              bbox_to_anchor=(-0.05, -0.32), ncol=2)
    fig.subplots_adjust(bottom=0.30)
    return fig, tA, pA


def plot_joint_stereonet(joint_sets):
    # stereonet kekar: mirip sesar tapi biasanya lebih banyak data
    # kekar = rekahan tanpa pergerakan, jadi ga ada rake/pitch
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5),
        subplot_kw=dict(projection='stereonet')
    )
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    legend_handles = []
    for i, js in enumerate(joint_sets):
        col = COLORS[i % len(COLORS)]
        ax.plane(js['strike'], js['dip'], color=col, linewidth=1.5, alpha=0.65)
        ax.pole(js['strike'],  js['dip'], color=col, marker='D',
                markersize=6)   # marker diamond (D) biar beda dari sesar (circle)
        legend_handles.append(
            Line2D([0],[0], color=col, linewidth=2,
                   label="J%d  %d°/%d°" % (i + 1, js['strike'], js['dip']))
        )

    ax.grid(True, alpha=0.3)
    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.set_azimuth_ticklabels(['N', 'E', 'S', 'W'], fontsize=10, fontweight='bold')
    ax.set_title("Stereonet Kekar\n(Great Circle & Pole)", fontsize=11, pad=20)
    ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5,
              framealpha=0.9, edgecolor='#ccc',
              bbox_to_anchor=(-0.05, -0.20), ncol=2)
    fig.subplots_adjust(bottom=0.22)
    return fig


def plot_joint_rose(strikes):
    # rose diagram khusus kekar — sama persis dengan rose sesar
    # dipisah fungsinya biar bisa dikustomisasi warnanya berbeda
    fig = plt.figure(figsize=(4.5, 4.5))
    ax  = fig.add_subplot(111, polar=True)
    fig.patch.set_facecolor('#FAFAF8')
    ax.set_facecolor('#FAFAF8')

    all_s = np.concatenate([strikes, (strikes + 180) % 360])
    bins  = np.linspace(0, 2 * np.pi, 37)
    n, _  = np.histogram(np.radians(all_s), bins=bins)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # warna hijau teal buat bedain dari rose sesar yang biru
    ax.bar(bins[:-1], n, width=2 * np.pi / 36,
           bottom=0, color='#1D9E75', edgecolor='white', alpha=0.8, linewidth=0.5)
    ax.set_yticks([])
    ax.set_title("Rose Diagram Kekar (Strike)", fontsize=11, pad=14)
    plt.tight_layout()
    return fig


def plot_joint_contour(joint_sets):
    # contour plot (density plot) buat lihat clustering pole kekar
    # berguna kalau data banyak dan mau lihat ada berapa set kekar
    # kamimplstereonet punya fungsi density() langsung
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5),
        subplot_kw=dict(projection='stereonet')
    )
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')

    strikes_arr = np.array([j['strike'] for j in joint_sets])
    dips_arr    = np.array([j['dip']    for j in joint_sets])

    # plot semua pole dulu sebagai titik
    for i, js in enumerate(joint_sets):
        col = COLORS[i % len(COLORS)]
        ax.pole(js['strike'], js['dip'], color=col, marker='D', markersize=5, alpha=0.7)

    # density contour kalau data >= 3 (minimal buat interpolasi)
    if len(joint_sets) >= 3:
        try:
            # cmap 'Greens' dipilih biar konsisten dengan tema warna kekar
            ax.density_contourf(strikes_arr, dips_arr, measurement='poles',
                                cmap='Greens', alpha=0.5)
            ax.density_contour(strikes_arr, dips_arr, measurement='poles',
                               colors='#0F6E56', linewidths=0.8, alpha=0.7)
        except Exception:
            pass   # kalau gagal (misal data terlalu sedikit), skip aja

    ax.grid(True, alpha=0.3)
    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.set_azimuth_ticklabels(['N', 'E', 'S', 'W'], fontsize=10, fontweight='bold')
    ax.set_title("Density Contour Pole Kekar", fontsize=11, pad=20)
    return fig


# ════════════════════════════════════════════════════════════
# SESSION STATE — nyimpen data input biar ga ilang tiap rerun
# ════════════════════════════════════════════════════════════
# streamlit rerun script dari atas tiap ada perubahan input
# session_state adalah cara simpan data yang persist antar rerun

if 'fault_sets' not in st.session_state:
    # contoh data sesar konjugat (dua kelompok strike berlawanan ~180°)
    st.session_state.fault_sets = [
        {'strike': 120, 'dip': 45, 'pitch': 0},
        {'strike': 130, 'dip': 50, 'pitch': 0},
        {'strike': 140, 'dip': 60, 'pitch': 0},
        {'strike': 300, 'dip': 30, 'pitch': 0},
        {'strike': 310, 'dip': 35, 'pitch': 0},
        {'strike': 320, 'dip': 40, 'pitch': 0},
    ]

if 'fold_left' not in st.session_state:
    st.session_state.fold_left = [
        {'strike': 120, 'dip': 40},
        {'strike': 125, 'dip': 45},
        {'strike': 130, 'dip': 50},
    ]

if 'fold_right' not in st.session_state:
    st.session_state.fold_right = [
        {'strike': 300, 'dip': 35},
        {'strike': 305, 'dip': 40},
        {'strike': 310, 'dip': 45},
    ]

if 'joint_sets' not in st.session_state:
    # contoh data kekar tiga set dengan orientasi berbeda
    st.session_state.joint_sets = [
        {'strike':  45, 'dip': 85},
        {'strike':  50, 'dip': 80},
        {'strike': 135, 'dip': 88},
        {'strike': 140, 'dip': 75},
        {'strike': 220, 'dip': 70},
        {'strike': 225, 'dip': 82},
    ]


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════

st.title("🪨 Stereonet Geologi Struktur")
st.caption("Proyeksi stereonet interaktif — Sesar, Lipatan & Kekar")
st.divider()

# pilihan halaman pakai radio horizontal di atas
mode = st.radio(
    "Pilih halaman analisis",
    ["⚡ Sesar (Fault)", "〰️ Lipatan (Fold)", "💎 Kekar (Joint)"],
    horizontal=True
)
st.divider()


# ════════════════════════════════════════════════════════════
# PAGE: ANALISIS SESAR
# ════════════════════════════════════════════════════════════

if mode.startswith("⚡"):

    col_input, col_plot = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Input Data Sesar")
        fs        = st.session_state.fault_sets
        to_delete = None

        for i, s in enumerate(fs):
            st.markdown(
                "<small style='color:#888'>Set %d</small>" % (i + 1),
                unsafe_allow_html=True
            )
            ca, cb, cc, cd = st.columns([2, 2, 2, 1])
            with ca:
                fs[i]['strike'] = st.number_input(
                    "Strike %d (°)" % (i + 1), 0, 360,
                    s['strike'], key="fs_%d_str" % i
                )
            with cb:
                fs[i]['dip'] = st.number_input(
                    "Dip %d (°)" % (i + 1), 0, 90,
                    s['dip'], key="fs_%d_dip" % i
                )
            with cc:
                # pitch = rake manual, opsional
                # kalau diisi 0 berarti akan dihitung otomatis
                fs[i]['pitch'] = st.number_input(
                    "Pitch %d (°)" % (i + 1), 0, 90,
                    s.get('pitch', 0), key="fs_%d_pit" % i
                )
            with cd:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fs) > 1:
                    if st.button("✕", key="del_fs_%d" % i, help="Hapus"):
                        to_delete = i
                st.markdown("</div>", unsafe_allow_html=True)

        # hapus dilakukan di luar loop biar index ga kacau
        if to_delete is not None:
            st.session_state.fault_sets.pop(to_delete)
            st.rerun()

        if st.button("＋ Tambah set data", use_container_width=True):
            st.session_state.fault_sets.append({'strike': 0, 'dip': 45, 'pitch': 0})
            st.rerun()

        st.divider()
        shear_sense = st.selectbox(
            "Shear sense",
            ["Dextral (kanan)", "Sinistral (kiri)"]
        )

    with col_plot:
        strikes = np.array([s['strike'] for s in fs])
        dips    = np.array([s['dip']    for s in fs])
        pitches = np.array([s.get('pitch', 0) for s in fs])

        mean_pitch = float(np.mean(pitches))
        # kalau pitch rata-rata > 0 berarti pengguna ngisi manual, pakai itu
        # kalau 0 semua, estimasi otomatis dari dip & shear sense
        rake = int(mean_pitch) if mean_pitch > 0 \
               else estimate_rake(float(np.mean(dips)), shear_sense)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Stereonet", "Rose Diagram", "Stress Axes", "Beachball"
        ])

        with tab1:
            fig = plot_fault_stereonet(fs)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)   # plt.close() penting buat free memory, kalau ga lama2 berat

        with tab2:
            fig = plot_rose(strikes)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab3:
            # σ1 dihitung tegak lurus rata-rata strike (Anderson's theory)
            sigma1 = (mean_angle(list(strikes)) + 90) % 360
            sigma2 = (sigma1 + 90)  % 360
            sigma3 = (sigma1 + 180) % 360
            fig    = plot_stress_stereonet(sigma1, sigma2, sigma3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab4:
            # beachball pakai rata-rata strike & dip semua set
            fig = plot_beachball(int(np.mean(strikes)), int(np.mean(dips)), rake)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if not OBSPY_OK:
                st.caption("⚠️ Untuk beachball akurat install: `pip install obspy`")

    st.divider()
    st.subheader("Hasil Analisis")

    # klasifikasi sesar — pakai rake kalau pitch sudah diisi, fallback dip kalau belum
    # tanda * di hasil berarti estimasi dari dip, bukan rake → kurang akurat
    if mean_pitch > 0:
        fault_type = classify_fault_by_rake(rake)
        rake_source = "dari pitch manual"
    else:
        fault_type = classify_fault_by_dip(list(dips))
        rake_source = "estimasi dari dip (isi pitch untuk hasil akurat)"

    sigma1 = (mean_angle(list(strikes)) + 90) % 360
    sigma3 = (sigma1 + 180) % 360

    # analisis Riedel shear: kelompokin tiap strike ke tipe geser
    # shear zone = bin dengan frekuensi strike terbanyak
    hist = np.zeros(18)   # 18 bin × 20° = 360°
    for s in strikes:
        hist[int(((s % 360) + 360) % 360 // 20) % 18] += 1
    shear_zone = float(np.argmax(hist) * 20 + 10)

    riedel_types = []
    for s in strikes:
        a = angular_diff(float(s), shear_zone)
        if a <= 10:          riedel_types.append('Y shear')
        elif a <= 20:        riedel_types.append('R shear')
        elif a <= 40:        riedel_types.append('P shear')
        elif 70 <= a <= 80:  riedel_types.append("R' shear")
        else:                riedel_types.append('Unclassified')

    rcount   = Counter(riedel_types)
    dominant = rcount.most_common(1)[0][0]

    m1, m2, m3, m4, m5 = st.columns(5)
    for col_m, lbl, val in [
        (m1, "Fault type",   fault_type),
        (m2, "σ1 trend",     "%.1f°" % sigma1),
        (m3, "σ3 trend",     "%.1f°" % sigma3),
        (m4, "Rake / Pitch", "%d°"   % rake),
        (m5, "Riedel dom.",  dominant),
    ]:
        with col_m:
            st.markdown("""
            <div class="metric-box">
              <div class="metric-label">%s</div>
              <div class="metric-value">%s</div>
            </div>""" % (lbl, val), unsafe_allow_html=True)

    # warning kalau klasifikasi masih dari dip
    if mean_pitch == 0:
        st.warning("⚠️ Pitch belum diisi — klasifikasi sesar pakai estimasi dari dip. Isi kolom Pitch untuk hasil yang akurat.", icon="⚠️")

    st.markdown("""
    <div class="info-box">
      <b>Riedel shear zone:</b> %.0f° &nbsp;|&nbsp;
      <b>Struktur dominan:</b> %s &nbsp;|&nbsp;
      <b>Shear sense:</b> %s<br>
      <b>Beachball input:</b> Strike %d° / Dip %d° / Rake %d° (%s)<br>
      <b>Distribusi Riedel:</b> %s
    </div>""" % (
        shear_zone, dominant, shear_sense,
        int(np.mean(strikes)), int(np.mean(dips)), rake, rake_source,
        " · ".join("%s: %d" % (k, v) for k, v in rcount.items())
    ), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE: ANALISIS LIPATAN
# ════════════════════════════════════════════════════════════

elif mode.startswith("〰"):

    col_input, col_plot = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Input Data Lipatan")

        st.markdown("**Limb kiri**")
        fl    = st.session_state.fold_left
        del_l = None

        for i, s in enumerate(fl):
            ca, cb, cc = st.columns([2, 2, 1])
            with ca:
                fl[i]['strike'] = st.number_input(
                    "Strike L%d" % (i + 1), 0, 360,
                    s['strike'], key="fl_%d_str" % i
                )
            with cb:
                fl[i]['dip'] = st.number_input(
                    "Dip L%d" % (i + 1), 0, 90,
                    s['dip'], key="fl_%d_dip" % i
                )
            with cc:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fl) > 1:
                    if st.button("✕", key="del_fl_%d" % i):
                        del_l = i
                st.markdown("</div>", unsafe_allow_html=True)

        if del_l is not None:
            st.session_state.fold_left.pop(del_l)
            st.rerun()

        if st.button("＋ Tambah limb kiri", use_container_width=True):
            st.session_state.fold_left.append({'strike': 0, 'dip': 45})
            st.rerun()

        st.divider()

        st.markdown("**Limb kanan**")
        fr    = st.session_state.fold_right
        del_r = None

        for i, s in enumerate(fr):
            ca, cb, cc = st.columns([2, 2, 1])
            with ca:
                fr[i]['strike'] = st.number_input(
                    "Strike R%d" % (i + 1), 0, 360,
                    s['strike'], key="fr_%d_str" % i
                )
            with cb:
                fr[i]['dip'] = st.number_input(
                    "Dip R%d" % (i + 1), 0, 90,
                    s['dip'], key="fr_%d_dip" % i
                )
            with cc:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(fr) > 1:
                    if st.button("✕", key="del_fr_%d" % i):
                        del_r = i
                st.markdown("</div>", unsafe_allow_html=True)

        if del_r is not None:
            st.session_state.fold_right.pop(del_r)
            st.rerun()

        if st.button("＋ Tambah limb kanan", use_container_width=True):
            st.session_state.fold_right.append({'strike': 0, 'dip': 45})
            st.rerun()

    with col_plot:
        fig_fold, tA, pA = plot_fold_stereonet(fl, fr)
        st.pyplot(fig_fold, use_container_width=True)
        plt.close(fig_fold)

    st.divider()
    st.subheader("Hasil Analisis Lipatan")

    # klasifikasi Fleuty lengkap: orientasi (dari plunge) + tightness (dari interlimb angle)
    plunge_class, tight_class = classify_fold_fleuty(pA)
    interlimb = calc_interlimb_angle(fl, fr)
    _, tight_class = classify_fold_fleuty(pA, interlimb)

    def mv_local(sets):
        vecs = [plane_to_vec(s['strike'], s['dip']) for s in sets]
        return normalize(np.mean(vecs, axis=0))

    mL = mv_local(fl)
    mR = mv_local(fr)
    tL, pL = vec_to_trend_plunge(mL)
    tR, pR = vec_to_trend_plunge(mR)

    # 6 metrik: axis trend, plunge, interlimb, orientasi Fleuty, tightness, mean pole
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)
    for col_m, lbl, val in [
        (m1, "Fold axis trend",        "%.1f°" % tA),
        (m2, "Fold axis plunge",       "%.1f°" % pA),
        (m3, "Interlimb angle",        "%.1f°" % interlimb),
        (m4, "Orientasi (Fleuty)",     plunge_class),
        (m5, "Tightness (Fleuty)",     tight_class),
        (m6, "Mean pole kiri / kanan", "%.0f°/%.0f°  ·  %.0f°/%.0f°" % (tL, pL, tR, pR)),
    ]:
        with col_m:
            st.markdown("""
            <div class="metric-box">
              <div class="metric-label">%s</div>
              <div class="metric-value" style="font-size:15px">%s</div>
            </div>""" % (lbl, val), unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <b>Klasifikasi Fleuty (1964):</b><br>
      &nbsp;&nbsp;Orientasi sumbu: <b>%s</b> (plunge %.1f°)<br>
      &nbsp;&nbsp;Kekencangan lipatan: <b>%s</b> (interlimb %.1f°)<br>
      <b>Simbol stereonet:</b>
        Lingkaran (●) = pole data &nbsp;|&nbsp;
        Segitiga (▲) = mean pole per limb &nbsp;|&nbsp;
        Bintang (★) = fold axis<br>
      <b>Cara baca interlimb angle:</b>
        sudut antara dua limb — makin kecil makin terlipat ketat.
        Dihitung dari sudut antar mean pole kedua limb.
    </div>""" % (plunge_class, pA, tight_class, interlimb), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE: ANALISIS KEKAR
# ════════════════════════════════════════════════════════════

else:

    col_input, col_plot = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("Input Data Kekar")
        # kekar ga punya arah slip, jadi cukup strike + dip aja
        # marker yang dipakai diamond (D) supaya beda dari sesar

        js        = st.session_state.joint_sets
        to_del_j  = None

        for i, s in enumerate(js):
            st.markdown(
                "<small style='color:#888'>Kekar J%d</small>" % (i + 1),
                unsafe_allow_html=True
            )
            ca, cb, cc = st.columns([2, 2, 1])
            with ca:
                js[i]['strike'] = st.number_input(
                    "Strike J%d (°)" % (i + 1), 0, 360,
                    s['strike'], key="js_%d_str" % i
                )
            with cb:
                js[i]['dip'] = st.number_input(
                    "Dip J%d (°)" % (i + 1), 0, 90,
                    s['dip'], key="js_%d_dip" % i
                )
            with cc:
                st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
                if len(js) > 1:
                    if st.button("✕", key="del_js_%d" % i, help="Hapus"):
                        to_del_j = i
                st.markdown("</div>", unsafe_allow_html=True)

        if to_del_j is not None:
            st.session_state.joint_sets.pop(to_del_j)
            st.rerun()

        if st.button("＋ Tambah data kekar", use_container_width=True):
            st.session_state.joint_sets.append({'strike': 0, 'dip': 90})
            st.rerun()

    with col_plot:
        strikes_j = np.array([j['strike'] for j in js])
        dips_j    = np.array([j['dip']    for j in js])

        # 3 tab: stereonet, rose, density contour
        tab1, tab2, tab3 = st.tabs(["Stereonet", "Rose Diagram", "Density Contour"])

        with tab1:
            fig = plot_joint_stereonet(js)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            fig = plot_joint_rose(strikes_j)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab3:
            # density contour berguna banget kalau data kekar banyak
            # bisa langsung kelihatan ada berapa cluster/set kekar
            fig = plot_joint_contour(js)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if len(js) < 3:
                st.caption("Tambah minimal 3 data untuk density contour")

    st.divider()
    st.subheader("Hasil Analisis Kekar")

    # klasifikasi tiap kekar satu per satu
    joint_classes  = [classify_joint(d) for d in dips_j]
    class_counter  = Counter(joint_classes)
    dominant_class = class_counter.most_common(1)[0][0]

    # sistem kekar berdasarkan orientasi antar-bidang
    joint_system = classify_joint_system(list(strikes_j))

    # rata-rata strike & dip kekar
    mean_strike_j = mean_angle(list(strikes_j))
    mean_dip_j    = float(np.mean(dips_j))

    m1, m2, m3, m4 = st.columns(4)
    for col_m, lbl, val in [
        (m1, "Tipe dominan",    dominant_class.split("(")[0].strip()),
        (m2, "Mean strike",     "%.1f°" % mean_strike_j),
        (m3, "Mean dip",        "%.1f°" % mean_dip_j),
        (m4, "Jumlah data",     "%d bidang" % len(js)),
    ]:
        with col_m:
            st.markdown("""
            <div class="metric-box">
              <div class="metric-label">%s</div>
              <div class="metric-value">%s</div>
            </div>""" % (lbl, val), unsafe_allow_html=True)

    # tabel distribusi tipe kekar
    st.markdown("**Distribusi tipe kekar:**")
    col_tbl = st.columns(min(len(class_counter), 4))
    for idx, (kls, cnt) in enumerate(class_counter.most_common()):
        with col_tbl[idx % len(col_tbl)]:
            st.markdown("""
            <div class="metric-box">
              <div class="metric-label">%s</div>
              <div class="metric-value">%d data</div>
            </div>""" % (kls.split("(")[0].strip(), cnt), unsafe_allow_html=True)

    # detail tiap bidang kekar
    with st.expander("Detail tiap bidang kekar"):
        st.markdown(
            "| No | Strike | Dip | Tipe |",
            unsafe_allow_html=False
        )
        st.markdown("| --- | --- | --- | --- |")
        for i, (s, d, c) in enumerate(zip(strikes_j, dips_j, joint_classes)):
            st.markdown("| J%d | %d° | %d° | %s |" % (i + 1, s, d, c))

    st.markdown("""
    <div class="info-box-green">
      <b>Sistem kekar:</b> %s<br>
      <b>Tipe dominan:</b> %s<br>
      <b>Mean orientasi:</b> Strike %.1f° / Dip %.1f°<br>
      <b>Simbol:</b> Diamond (◆) = pole kekar di stereonet<br>
      <b>Catatan:</b> Density contour membantu identifikasi jumlah set kekar
      dari clustering distribusi pole.
    </div>""" % (
        joint_system, dominant_class,
        mean_strike_j, mean_dip_j
    ), unsafe_allow_html=True)


# ── footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "Stereonet Geologi Struktur  ·  Proyeksi Wulff (Equal-Angle)  ·  Lower Hemisphere"
)
