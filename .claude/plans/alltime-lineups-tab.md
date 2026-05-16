# All-Time Lineups Tab (Phase 2)

## Context
The app has a "Single Season" tab for current-season predictions. The "All-Time Lineups" tab (currently a placeholder) lets users build fantasy lineups from any player in MLB history — each player selected from their own specific year. The model's feature vector (BA, OBP, SLG, ERA, etc.) is unchanged, so `/predict` is shared between both tabs.

---

## Step 1: Build the player-season dataset — `app/build_player_seasons.py`

A one-time script (re-run each offseason to add new data) that generates `app/data/player_seasons.json`.

**Source:** pybaseball's Lahman database via `pybaseball.batting()` and `pybaseball.pitching()`

**Filters:**
- Batters: PA >= 100
- Pitchers: IP >= 30

**Output format** — a dict keyed by `"Player Name (YYYY)"` with stats as the value:
```json
{
  "Babe Ruth (1927)": {"kind": "batter", "BA": 0.356, "OBP": 0.486, "SLG": 0.772, "OPS": 1.258, "K%": 0.08, "BB%": 0.20},
  "Sandy Koufax (1966)": {"kind": "pitcher", "ERA": 1.73, "WHIP": 0.969, "SO9": 10.55, "SO/W": 4.55, "IP": 323.0},
  ...
}
```

**Why this format:**
- Single file load → both autocomplete list and stats available client-side immediately
- No per-player API call needed at runtime — stats fill instantly on selection
- Estimated size: ~30k entries × ~80 bytes ≈ 2.4MB uncompressed, ~400KB gzipped

**Stat column mapping from pybaseball:**

| App stat | pybaseball batting col | pybaseball pitching col |
|---|---|---|
| BA | `AVG` | — |
| OBP | `OBP` | — |
| SLG | `SLG` | — |
| OPS | `OPS` | — |
| K% | `SO%` (or `SO/PA`) | — |
| BB% | `BB%` (or `BB/PA`) | — |
| ERA | — | `ERA` |
| WHIP | — | `WHIP` |
| SO9 | — | `K/9` |
| SO/W | — | `K/BB` |
| IP | — | `IP` |

Verify column names against actual pybaseball output before finalizing.

---

## Step 2: Backend additions — `app/app.py`

Load `player_seasons.json` at startup (analogous to `MODEL`):
```python
import json

_ALLTIME_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "player_seasons.json")
try:
    with open(_ALLTIME_DATA_PATH) as f:
        PLAYER_SEASONS = json.load(f)
except FileNotFoundError:
    PLAYER_SEASONS = {}
```

New endpoint:
```python
@app.route("/data/player_seasons")
def player_seasons():
    return jsonify(PLAYER_SEASONS)
```

The `/predict` endpoint remains **unchanged**.

---

## Step 3: All-Time form — `app/templates/index.html`

Replace the `#tab-alltime` placeholder with a full form. The form mirrors the Single Season layout with these differences:

- **No season selector** — year is encoded in each player name (`"Babe Ruth (1927)"`)
- **Stats fill automatically on player selection** — no "Auto-fill Stats" button needed
- **Keep manual toggle** as a fallback for custom stat entry
- **Session storage key:** `"lineup_form_alltime"` (separate from `"lineup_form_current"`)
- **Form id:** `alltime-form`, POST to `/predict`

Enable the tab button (remove `disabled` attribute) once the form is in place.

---

## Step 4: Custom autocomplete widget — `app/templates/index.html`

The all-time player list (~30k entries) exceeds `<datalist>` performance limits, so use a custom dropdown. The Single Season tab's `setupAutocomplete()` can be reused with one extension: after a player is selected, parse the year from `"Name (YYYY)"` and fill stats from the in-memory dict.

**Data flow on player selection:**
1. User types "Babe Ruth" → dropdown shows `"Babe Ruth (1927)"`, `"Babe Ruth (1926)"`, etc.
2. User clicks `"Babe Ruth (1927)"`
3. JS sets input value, then looks up `PLAYER_SEASONS["Babe Ruth (1927)"]`
4. Fills stat fields for that slot from the returned dict

**JS additions** (inside existing `<script>` block):
```javascript
let playerSeasons = {};  // loaded once

fetch("/data/player_seasons")
  .then(r => r.ok ? r.json() : {})
  .then(data => { playerSeasons = data; })
  .catch(() => {});

function setupAlltimeAutocomplete(input, slotPrefix) {
  const names = Object.keys(playerSeasons);
  const dropdown = document.createElement("div");
  dropdown.className = "autocomplete-dropdown";
  dropdown.hidden = true;
  input.parentElement.appendChild(dropdown);

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 3) { dropdown.hidden = true; return; }
    const matches = names.filter(n => n.toLowerCase().includes(q)).slice(0, 5);
    // render matches... (same pattern as setupAutocomplete)
  });

  // On selection: fill stats
  function selectPlayer(name) {
    input.value = name;
    dropdown.hidden = true;
    const stats = playerSeasons[name];
    if (!stats) return;
    Object.entries(stats).forEach(([stat, val]) => {
      if (stat === "kind") return;
      set(`${slotPrefix}_${stat}`, val);
    });
  }
}
```

Apply to all `.player-name-input` elements inside `#tab-alltime`, passing each input its slot prefix (e.g., `"home_sp"`, `"away_bat3"`).

---

## Critical files

| File | Change |
|---|---|
| `app/build_player_seasons.py` | New script — generates `player_seasons.json` |
| `app/data/player_seasons.json` | Generated output (gitignore this file) |
| `app/app.py` | Load `PLAYER_SEASONS`, add `/data/player_seasons` endpoint |
| `app/templates/index.html` | Replace placeholder with full all-time form + autocomplete |

---

## Verification

1. Run `python build_player_seasons.py` → `app/data/player_seasons.json` generated, key count reasonable (20k–40k)
2. `GET /data/player_seasons` returns the dict with correct structure
3. Remove `disabled` from "All-Time Lineups" tab button, confirm tab switches correctly
4. Type "Sandy Ko" → dropdown shows "Sandy Koufax (1966)", "Sandy Koufax (1965)", etc.
5. Select a player → stat fields fill immediately without any button click
6. Build a full 2-team lineup and submit → `/predict` returns a result
7. Session storage for all-time tab is independent of single season tab
