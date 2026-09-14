/* Shared sign-in for every SAHAi page.
 *
 * The three pages used to carry a free-text "learner" box whose value went out
 * as `x-learner-id`, which meant typing someone else's id was enough to read
 * their sessions, mastery and submissions. The gateway now requires a bearer
 * token, so that box has no meaning: this replaces it with an identity chip and
 * a sign-in gate.
 *
 * The token lives in localStorage. That is readable by any script on this
 * origin, so it is worth being plain about the trade: an httpOnly cookie would
 * be safer against XSS, and would need CSRF protection and a same-origin
 * assumption this stack does not otherwise make. For a single-origin static
 * page with no third-party scripts, localStorage is the honest simple choice —
 * and the token is a bearer credential either way.
 */

const SAHAI_TOKEN_KEY = 'sahai_token';

function authToken() {
  try { return localStorage.getItem(SAHAI_TOKEN_KEY) || ''; } catch (_) { return ''; }
}

function setAuthToken(token) {
  try {
    if (token) localStorage.setItem(SAHAI_TOKEN_KEY, token);
    else localStorage.removeItem(SAHAI_TOKEN_KEY);
  } catch (_) { /* private mode — the session still works, it just won't persist */ }
}

function authHeaders() {
  const t = authToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function signOut() {
  setAuthToken('');
  // Full reload rather than re-rendering: every page holds per-learner state
  // in module scope (selected problem, open session, mastery panel) and
  // unpicking all of it by hand is how the old learner-switcher leaked one
  // learner's turns into another's transcript.
  location.reload();
}

/* ------------------------------------------------------------------ gate */

const AUTH_CSS = `
#sahai-auth { position: fixed; inset: 0; z-index: 9999; display: flex;
  align-items: center; justify-content: center; background: var(--ink, #0d1030);
  background-image: radial-gradient(circle at 30% 20%, rgba(249,97,103,.10), transparent 55%); }
#sahai-auth .card { width: min(420px, calc(100vw - 32px)); padding: 28px;
  border: 1px solid var(--rule, #272c66); border-radius: 14px;
  background: var(--ink-2, #141845); }
#sahai-auth h1 { font-family: var(--display, Georgia, serif); font-size: 26px;
  margin: 0 0 6px; color: var(--cream, #f2ece1); }
#sahai-auth p { margin: 0 0 18px; color: var(--cream-dim, #9a99b8); font-size: 13px; }
#sahai-auth label { display: block; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--cream-dim, #9a99b8); margin: 0 0 6px; }
#sahai-auth input { width: 100%; padding: 10px 12px; border-radius: 8px;
  border: 1px solid var(--rule, #272c66); background: var(--ink, #0d1030);
  color: var(--cream, #f2ece1); font: inherit; margin-bottom: 14px; }
#sahai-auth button { width: 100%; padding: 11px; border: 0; border-radius: 8px;
  background: var(--pink, #f96167); color: #fff; font: inherit; font-weight: 600;
  cursor: pointer; }
#sahai-auth button[disabled] { opacity: .5; cursor: default; }
#sahai-auth .alt { margin-top: 16px; text-align: center; font-size: 13px; }
#sahai-auth .alt a { color: var(--pink, #f96167); cursor: pointer; text-decoration: none; }
#sahai-auth .err { color: var(--bad, #f96167); font-size: 13px; margin-bottom: 12px; }
#sahai-auth .token { font-family: var(--mono, monospace); font-size: 12px;
  word-break: break-all; padding: 12px; border-radius: 8px;
  border: 1px dashed var(--pink-dim, #7a2f41); background: var(--ink, #0d1030);
  color: var(--cream, #f2ece1); margin-bottom: 14px; }
#sahai-auth .warn { color: var(--pink, #f96167); font-size: 12px; margin-bottom: 14px; }
.sahai-whoami { display: flex; align-items: center; gap: 8px; font-size: 13px;
  color: var(--cream-dim, #9a99b8); }
.sahai-whoami strong { color: var(--cream, #f2ece1); font-weight: 600; }
.sahai-whoami a { color: var(--pink, #f96167); cursor: pointer; text-decoration: none;
  font-size: 12px; }
`;

function mountAuthStyles() {
  if (document.getElementById('sahai-auth-css')) return;
  const style = document.createElement('style');
  style.id = 'sahai-auth-css';
  style.textContent = AUTH_CSS;
  document.head.appendChild(style);
}

function authGate() {
  mountAuthStyles();
  const host = document.createElement('div');
  host.id = 'sahai-auth';
  document.body.appendChild(host);

  const render = (mode, error) => {
    host.innerHTML = mode === 'token'
      ? `<div class="card">
           <h1>Sign in</h1>
           <p>Paste the token you saved when you registered.</p>
           ${error ? `<div class="err">${error}</div>` : ''}
           <label for="sa-token">Token</label>
           <input id="sa-token" autocomplete="off" spellcheck="false">
           <button id="sa-go">Continue</button>
           <div class="alt">No token yet? <a id="sa-swap">Create a learner</a></div>
         </div>`
      : `<div class="card">
           <h1>SAHAi</h1>
           <p>A name to show on your progress page. No email, no password.</p>
           ${error ? `<div class="err">${error}</div>` : ''}
           <label for="sa-name">What should we call you?</label>
           <input id="sa-name" autocomplete="off" maxlength="80">
           <button id="sa-go">Create a learner</button>
           <div class="alt">Already have a token? <a id="sa-swap">Sign in</a></div>
         </div>`;

    const field = document.getElementById(mode === 'token' ? 'sa-token' : 'sa-name');
    const go = document.getElementById('sa-go');
    field.focus();
    field.addEventListener('keydown', e => { if (e.key === 'Enter') go.click(); });
    document.getElementById('sa-swap').addEventListener(
      'click', () => render(mode === 'token' ? 'register' : 'token'),
    );

    go.addEventListener('click', async () => {
      const value = field.value.trim();
      if (!value) { field.focus(); return; }
      go.disabled = true;
      try {
        if (mode === 'token') {
          const res = await fetch('/v1/auth/me', {
            headers: { Authorization: `Bearer ${value}` },
          });
          if (!res.ok) throw new Error('That token was not recognised.');
          setAuthToken(value);
          location.reload();
        } else {
          const res = await fetch('/v1/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ display_name: value }),
          });
          if (res.status === 429) throw new Error('Too many registrations from here. Try later.');
          if (!res.ok) throw new Error('Could not create a learner.');
          const body = await res.json();
          setAuthToken(body.token);
          showTokenOnce(host, body);
        }
      } catch (err) {
        go.disabled = false;
        render(mode, err.message);
      }
    });
  };

  render(authToken() ? 'token' : 'register');
}

function showTokenOnce(host, body) {
  // The one screen where the token is visible. The server stores only a hash,
  // so if this is dismissed without saving it there is no way to get it back
  // short of an operator running scripts/mint_token.py.
  host.innerHTML = `
    <div class="card">
      <h1>Save this token</h1>
      <p>It is your only way back in on another browser or device.</p>
      <div class="token" id="sa-tok">${body.token}</div>
      <div class="warn">It is not shown again and cannot be recovered.</div>
      <button id="sa-copy">Copy and continue</button>
      <div class="alt"><a id="sa-skip">I've saved it — continue</a></div>
    </div>`;
  const done = () => location.reload();
  document.getElementById('sa-skip').addEventListener('click', done);
  document.getElementById('sa-copy').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(body.token); } catch (_) { /* http, or denied */ }
    done();
  });
}

/* Renders "signed in as X · sign out" into `el`, or opens the gate. Every page
 * calls this before its first request, so an expired or deleted token is
 * caught once here rather than as a 401 per panel. */
async function requireAuth(el) {
  if (!authToken()) { authGate(); return null; }
  let me;
  try {
    const res = await fetch('/v1/auth/me', { headers: authHeaders() });
    if (!res.ok) throw new Error('unauthenticated');
    me = await res.json();
  } catch (_) {
    setAuthToken('');
    authGate();
    return null;
  }
  if (el) {
    el.className = 'sahai-whoami';
    el.innerHTML = `<strong></strong><a>sign out</a>`;
    el.querySelector('strong').textContent = me.display_name;
    el.querySelector('a').addEventListener('click', signOut);
    el.title = me.learner_id;
  }
  return me;
}
