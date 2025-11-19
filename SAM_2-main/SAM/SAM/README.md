# 🎾 SAM - Tennis Court Analysis Module

## Overzicht
Deze module visualiseert tennis data op een realistische tennisbaan met behulp van Plotly.

## Bestanden
- **tennis_court_draw.py**: Standalone module om een tennisbaan te tekenen
- **N_VA_TennisCourt.py**: Visualisatie module die trajecten van spelers/bal toont op de tennisbaan

## Installatie

```bash
pip install plotly
pip install kaleido  # Optioneel, voor PNG export
```

## Gebruik

### Standalone Tennisbaan
```python
from tennis_court_draw import create_tennis_court

# Maak een lege tennisbaan
fig = create_tennis_court()
fig.show()

# Voeg eigen data toe
fig.add_trace(go.Scatter(
    x=[4.0, 5.0], 
    y=[10.0, 15.0],
    mode='lines+markers',
    name='Traject'
))
fig.show()
```

### Tennis Visualisatie Module
De `N_VA_TennisCourt.py` module werkt net als andere N_VA modules in het PDP framework:

1. Laadt dataset via `av.dataset_name`
2. Gebruikt `av.poi` (aantal punten) en `av.tst` (aantal timestamps)
3. Maakt visualisaties per configuratie
4. Slaat HTML (en optioneel PNG) op in `AV_RESULTS_DIR/TennisCourt/`

## Tennisbaan Specificaties

Alle afmetingen zijn conform officiële ITF regels:

| Element | Afmeting |
|---------|----------|
| Singles breedte | 8.23 m |
| Doubles breedte | 10.97 m |
| Totale lengte | 23.77 m |
| Service box diepte | 6.40 m |
| Net positie | 11.885 m (midden) |
| Doubles alley | 1.37 m (per kant) |

### Coördinaten Systeem
- **Oorsprong (0, 0)**: Linksonder van singles baan
- **X-as**: Breedte (0 tot 8.23m voor singles)
- **Y-as**: Lengte (0 tot 23.77m)
- **Doubles alleys**: Negatieve x-waarden links, >8.23m rechts

## Kleurenschema

### Voor 2 punten (2 spelers)
- **Geel**: Speler 1
- **Cyaan**: Speler 2

### Voor 3 punten (2 spelers + bal)
- **Geel**: Speler 1
- **Cyaan**: Speler 2
- **Wit**: Bal

### Achtergrond
- **Baan**: Groen (#25D366) - grasbaan
- **Lijnen**: Wit
- **Papier achtergrond**: Donkergrijs (#1a1a1a)

## Visualisatie Features

### Enkele Timestamp
- Toont punten op hun posities
- Labels bij elk punt (p0, p1, etc.)
- Hover info met coördinaten

### Meerdere Timestamps
- **Trajectlijnen**: Continue lijn per punt
- **Pijlen**: Richting van beweging
- **Positie markers**: Punt bij elke timestamp
- **Interactief**: Zoom, pan, hover details
- **Labels**: Alleen bij eerste positie

## Output

### HTML Bestanden
Interactieve Plotly visualisaties:
- Zoom en pan functionaliteit
- Hover voor details
- Download als PNG via interface
- Locatie: `{AV_RESULTS_DIR}/TennisCourt/Tennis_Config_{N}.html`

### PNG Bestanden (optioneel)
Statische afbeeldingen (vereist kaleido):
- Resolutie: 800x1400 pixels
- Locatie: `{AV_RESULTS_DIR}/TennisCourt/Tennis_Config_{N}.png`

## Integratie met PDP Framework

Om deze module te gebruiken in het PDP framework:

1. **Voeg toe aan av.py**:
```python
N_VA_TennisCourt = 1  # Enable tennis court visualization
```

2. **Voeg toe aan N_Moving_Objects.py**:
```python
if av.N_VA_TennisCourt == 1:
    import N_VA_TennisCourt
```

3. **Update GUI.py** (optioneel):
Voeg checkbox toe in visualization section:
```python
{'label': ' 🎾 Tennis Court', 'value': 'N_VA_TennisCourt'}
```

## Voorbeeld Data Formaat

CSV met 5 kolommen (geen header):
```
conID, tstID, poiID, x, y
0, 0, 0, 2.0, 5.0
0, 0, 1, 6.0, 18.0
0, 1, 0, 3.5, 8.0
0, 1, 1, 5.0, 15.0
```

## Aanpassingen

### Kleuren wijzigen
In `N_VA_TennisCourt.py`, regel 26-30:
```python
if av.poi == 2:
    colors = ['yellow', 'cyan']
    labels = ['Jouw label 1', 'Jouw label 2']
```

### Baan kleur wijzigen
In `tennis_court_draw.py`, regel 161:
```python
plot_bgcolor='#25D366',  # Verander deze hex code
```

Voorbeelden:
- **Clay court (gravel)**: `'#E07020'`
- **Hard court (blauw)**: `'#4682B4'`
- **Indoor (grijs)**: `'#808080'`

### Lijndikte aanpassen
In `tennis_court_draw.py`, zoek naar `line=dict(...)` en wijzig `width=` parameter.

## Troubleshooting

### "Module 'tennis_court_draw' not found"
Zorg dat `tennis_court_draw.py` in dezelfde map staat als `N_VA_TennisCourt.py`, of voeg de SAM folder toe aan Python path.

### PNG export werkt niet
Installeer kaleido:
```bash
pip install kaleido
```

### Punten vallen buiten de baan
Controleer dat x-waarden binnen 0-8.23m zijn en y-waarden binnen 0-23.77m (voor singles).

## Toekomstige Uitbreidingen

Mogelijke features:
- [ ] Heatmap overlay (speler posities)
- [ ] Rally animatie (frame-by-frame)
- [ ] Serve/return zones markeren
- [ ] Shot type annotaties
- [ ] Snelheid vectors
- [ ] Court coverage statistieken

## Contact

Voor vragen of suggesties, zie het hoofdproject README.
