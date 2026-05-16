# Plan: Import Game Modal

## Context

Replace the existing "Today's Game" dropdown in the matchup bar with an "Import Game" button that opens a modal. The modal handles both today's games and any past date. For Final games, the actual score is stored in hidden form fields and displayed on the results page so users can compare prediction vs. reality. This is the TODO item in `app/README.md`.

---

## User-confirmed decisions

- Modal (not inline controls) replaces the Today's Game dropdown
- Show Live + Final for today; Final only for past dates
- Show game labels in modal as "Final" / "🔴 Live" — **no scores visible until after predicting**
- Import auto-sets the `#season` selector to the game's year
- Actual score shown on result page only after prediction is submitted

---

## Files to Modify

1. `app/app.py`
2. `app/templates/index.html`
3. `app/templates/result.html`

---

## Step-by-Step Implementation

### 1. `app/app.py`

**A. Replace `/games/today` with `/games/date/<date_str>`**

The existing endpoint only serves today. Replace it (updating the single JS callsite) with one that accepts any date:

```python
@app.route("/games/date/<date_str>")
def games_by_date(date_str):
    # date_str from HTML date input: YYYY-MM-DD
    dt = datetime.date.fromisoformat(date_str)
    formatted = dt.strftime("%m/%d/%Y")
    is_today = dt == datetime.date.today()
    games = mlb.get_scheduled_games_by_date(formatted)
    result = []
    for g in games:
        state = g.status.abstract_game_state
        if state not in ("Live", "Final"):
            continue
        if not is_today and state != "Final":
            continue
        result.append({
            "game_pk": g.game_pk,
            "status": state,
            "away": g.teams.away.team.name,
            "home": g.teams.home.team.name,
            "game_year": dt.year,
        })
    return jsonify(result)
```

**B. Enhance `/game/<int:game_pk>/lineup`** to also return actual score when requested

Add an optional `?include_score=true` query param. The frontend passes this only for Final games, avoiding an extra API call for Live games:

```python
@app.route("/game/<int:game_pk>/lineup")
def game_lineup(game_pk):
    box = mlb.get_game_box_score(game_pk)

    def resolve(team):
        batters = []
        for pid in team.batting_order:
            p = team.players.get(f"ID{pid}")
            batters.append({"id": pid, "name": p.person.full_name if p else ""})
        sp_id = team.pitchers[0] if team.pitchers else None
        sp_player = team.players.get(f"ID{sp_id}") if sp_id else None
        return {
            "team": team.team.name,
            "sp": {"id": sp_id, "name": sp_player.person.full_name if sp_player else ""},
            "batters": batters,
        }

    payload = {"away": resolve(box.teams.away), "home": resolve(box.teams.home)}

    if request.args.get("include_score") == "true":
        try:
            linescore = mlb._mlb_adapter_v1.get(
                endpoint=f"game/{game_pk}/linescore", ep_params={}
            ).data
            payload["home_runs"] = linescore.get("teams", {}).get("home", {}).get("runs")
            payload["away_runs"] = linescore.get("teams", {}).get("away", {}).get("runs")
        except Exception:
            payload["home_runs"] = None
            payload["away_runs"] = None

    return jsonify(payload)
```

**C. Enhance `/predict`** to accept and pass actual score to the template

```python
@app.route("/predict", methods=["POST"])
def predict():
    if not validate_form(request.form):
        return redirect(url_for("index", error="Please fill in all stat fields."))

    home_team = request.form.get("home_team") or "Home"
    away_team = request.form.get("away_team") or "Away"
    if home_team == away_team:
        home_team += " (Home)"
        away_team += " (Away)"

    # Parse optional actual score from hidden fields
    actual_away_runs = request.form.get("actual_away_runs", type=int)
    actual_home_runs = request.form.get("actual_home_runs", type=int)

    try:
        features = parse_features(request.form)
        result = run_model(features)
    except Exception:
        return redirect(url_for("index", error="Prediction failed. Please check your inputs and try again."))

    home_win_prob = result["home_win_prob"]
    winner = home_team if home_win_prob >= 0.5 else away_team
    winner_win_prob = home_win_prob if winner == home_team else 1 - home_win_prob

    prediction_correct = None
    if actual_away_runs is not None and actual_home_runs is not None:
        actual_home_won = actual_home_runs > actual_away_runs
        model_home_won = home_win_prob >= 0.5
        prediction_correct = actual_home_won == model_home_won

    return render_template(
        "result.html",
        home_team=home_team,
        away_team=away_team,
        winner=winner,
        winner_win_prob=winner_win_prob,
        actual_away_runs=actual_away_runs,
        actual_home_runs=actual_home_runs,
        prediction_correct=prediction_correct,
    )
```

---

### 2. `app/templates/index.html`

**A. Replace the Today's Game picker in the matchup bar**

Remove:
```html
<span class="vs">·</span>
<label for="game-picker">Today's Game</label>
<select id="game-picker">
  <option value="">Loading…</option>
</select>
```

Add:
```html
<span class="vs">·</span>
<button type="button" id="import-game-btn" class="import-btn">Import Game</button>
```

**B. Add hidden fields to the form** (just before `</form>`):

```html
<input type="hidden" name="actual_away_runs" id="actual_away_runs_input">
<input type="hidden" name="actual_home_runs" id="actual_home_runs_input">
```

**C. Add the modal HTML** (just before `</body>`):

```html
<div id="import-modal-overlay" class="modal-overlay" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-header">
      <span id="modal-title" style="font-weight:600;">Import a Game</span>
      <button type="button" id="modal-close-btn" aria-label="Close">×</button>
    </div>
    <div class="modal-body">
      <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1rem;">
        <label for="modal-date-picker" style="font-size:0.9rem; color:#555;">Date</label>
        <input type="date" id="modal-date-picker"
          style="padding:0.3rem 0.5rem; border:1px solid #ccc; border-radius:4px; font-size:0.9rem;">
      </div>
      <div id="modal-games-status" style="color:#888; font-size:0.85rem; margin-bottom:0.5rem;"></div>
      <div id="modal-games-list" style="display:flex; flex-direction:column; gap:0.4rem; max-height:280px; overflow-y:auto;"></div>
    </div>
  </div>
</div>
```

**D. Add modal CSS** (inside `<style>`):

```css
.import-btn {
  background: #1a3055;
  border: none;
  border-radius: 4px;
  color: #fff;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 500;
  padding: 0.35rem 0.8rem;
}
.import-btn:hover { background: #254470; }

.modal-overlay {
  align-items: center;
  background: rgba(0,0,0,0.35);
  bottom: 0;
  display: flex;
  justify-content: center;
  left: 0;
  position: fixed;
  right: 0;
  top: 0;
  z-index: 100;
}
.modal-overlay[hidden] { display: none; }
.modal-card {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.18);
  max-width: 420px;
  padding: 1.5rem;
  width: 90%;
}
.modal-header {
  align-items: center;
  display: flex;
  justify-content: space-between;
  margin-bottom: 1rem;
}
.modal-header button {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  font-size: 1.4rem;
  line-height: 1;
  padding: 0;
}
.modal-header button:hover { color: #333; }
.modal-game-btn {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  font-size: 0.9rem;
  justify-content: space-between;
  padding: 0.5rem 0.75rem;
  text-align: left;
  width: 100%;
}
.modal-game-btn:hover { background: #f0f4f8; }
.status-badge {
  font-size: 0.75rem;
  font-weight: 600;
  border-radius: 3px;
  padding: 0.15rem 0.4rem;
}
.status-live   { background: #ffeaea; color: #c00; }
.status-final  { background: #eaf2ea; color: #2a6e2a; }
```

**E. Update JavaScript** — replace `loadTodaysGames()` and the game-picker change handler with modal logic:

```javascript
// ── Import Game Modal ─────────────────────────────────────────

const modalOverlay   = document.getElementById("import-modal-overlay");
const modalDateInput = document.getElementById("modal-date-picker");
const modalGamesList = document.getElementById("modal-games-list");
const modalStatus    = document.getElementById("modal-games-status");

// Set date picker default to today, max to today
(function () {
  const today = new Date().toISOString().split("T")[0];
  modalDateInput.value = today;
  modalDateInput.max   = today;
})();

function openModal() {
  modalOverlay.hidden = false;
  loadGamesForDate(modalDateInput.value);
}

function closeModal() {
  modalOverlay.hidden = true;
}

document.getElementById("import-game-btn").addEventListener("click", openModal);
document.getElementById("modal-close-btn").addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

modalDateInput.addEventListener("change", () => {
  loadGamesForDate(modalDateInput.value);
});

async function loadGamesForDate(dateStr) {
  if (!dateStr) return;
  modalStatus.textContent = "Loading…";
  modalGamesList.innerHTML = "";

  const games = await fetch(`/games/date/${dateStr}`).then(r => r.json()).catch(() => []);

  if (!games.length) {
    modalStatus.textContent = "No games on this date.";
    return;
  }
  modalStatus.textContent = "";

  games.forEach(g => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "modal-game-btn";

    const live = g.status === "Live";
    btn.innerHTML = `
      <span>${g.away} @ ${g.home}</span>
      <span class="status-badge ${live ? "status-live" : "status-final"}">
        ${live ? "🔴 Live" : "Final"}
      </span>`;

    btn.addEventListener("click", () => importGame(g));
    modalGamesList.appendChild(btn);
  });
}

async function importGame(g) {
  closeModal();

  // Clear form
  form.querySelectorAll(".player-name-input").forEach(el => el.value = "");
  form.querySelectorAll('input[type="number"]').forEach(el => el.value = "");
  document.getElementById("actual_away_runs_input").value = "";
  document.getElementById("actual_home_runs_input").value = "";

  const isFinal = g.status === "Final";
  const url = `/game/${g.game_pk}/lineup${isFinal ? "?include_score=true" : ""}`;
  const data = await fetch(url).then(r => r.json()).catch(() => null);
  if (!data) return;

  // Populate team names
  set_input("away_team", data.away.team);
  set_input("home_team", data.home.team);

  // Populate player names
  ["away", "home"].forEach(side => {
    const ld = data[side];
    set_input(`${side}_sp_player_name`, ld.sp.name);
    ld.batters.forEach((b, i) => set_input(`${side}_bat${i + 1}_player_name`, b.name));
  });

  // Store actual score (Final games only)
  if (isFinal && data.home_runs != null && data.away_runs != null) {
    document.getElementById("actual_home_runs_input").value = data.home_runs;
    document.getElementById("actual_away_runs_input").value = data.away_runs;
  }

  // Set season to game year
  const seasonEl = document.getElementById("season");
  if (seasonEl) seasonEl.value = g.game_year;

  // Auto-fill stats
  const players = [];
  ["away", "home"].forEach(side => {
    const ld = data[side];
    players.push({ slot: `${side}_sp`, name: ld.sp.name, kind: "pitcher", player_id: ld.sp.id });
    ld.batters.forEach((b, i) => players.push({
      slot: `${side}_bat${i + 1}`, name: b.name, kind: "batter", player_id: b.id,
    }));
  });

  setFetching(true);
  try {
    const stats = await fetch("/autofill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ season: g.game_year, players }),
    }).then(r => r.json());
    Object.entries(stats).forEach(([field, val]) => set(field, val));
  } catch (e) {
    console.error("Import autofill failed:", e);
  }
  setFetching(false);
}
```

Remove the existing `loadTodaysGames()` call and the `document.getElementById("game-picker").addEventListener(...)` block entirely.

---

### 3. `app/templates/result.html`

Add a second card after the prediction card (before `</body>`):

```html
{% if actual_away_runs is not none and actual_home_runs is not none %}
<div class="card">
  <div class="result-label">Actual Result</div>
  <div class="actual-score">
    {{ away_team }} <span class="score-num">{{ actual_away_runs }}</span>
    &ndash;
    <span class="score-num">{{ actual_home_runs }}</span> {{ home_team }}
  </div>
  {% if prediction_correct %}
  <div class="verdict correct">✓ Correct prediction</div>
  {% else %}
  <div class="verdict incorrect">✗ Incorrect prediction</div>
  {% endif %}
</div>
{% endif %}
```

Add styles inside `<style>`:

```css
.actual-score {
  font-size: 1.4rem;
  font-weight: 700;
  color: #222;
  margin: 0.6rem 0 0.4rem;
}
.score-num { color: #1a3055; }
.verdict {
  font-size: 0.9rem;
  font-weight: 600;
  margin-top: 0.3rem;
}
.verdict.correct   { color: #2a7a2a; }
.verdict.incorrect { color: #b03030; }
```

---

## One Field-Verification Risk

The linescore path `teams.home.runs` / `teams.away.runs` is the expected MLB Stats API shape, but needs to be confirmed against a live API response during implementation. The try/except in the lineup endpoint absorbs any mismatch gracefully.

---

## Verification

1. `cd app && python app.py`
2. Click "Import Game" — confirm modal opens with today's date and today's games
3. Pick a past date — confirm only Final games appear
4. Click a Final game — confirm modal closes, team names + player names populate, stats autofill, season selector updates to game year
5. Click Predict — confirm result page shows prediction card + actual score card with correct/incorrect verdict
6. Click a Live game (today) — confirm modal closes, form populates, but **no actual score card** on result page
7. Click Predict without importing — confirm hidden score fields are empty, no actual score card shown
