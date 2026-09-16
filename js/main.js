/* ==========================================================================
   ROMAN'S PERSONAL WEBSITE - QUAKE TERMINAL EDITION
   Pure Vanilla JavaScript ES6+ | Zero Frameworks | Interactive Console & Game
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Modules
  initTopSystemBar();
  initBackgroundCanvas();
  initTerminalConsole();
  initNavigation();
  initAcornDatabase();
  initClickerGame();
  initTransmissionForm();
  initWebAudioFX();
});

/* ==========================================================================
   1. TOP SYSTEM BAR & AUDIO/CRT TOGGLES
   ========================================================================== */
let audioEnabled = true;

function initTopSystemBar() {
  // Real-time clock update
  const timeEl = document.getElementById('sys-time');
  function updateTime() {
    if (timeEl) {
      const now = new Date();
      timeEl.textContent = now.toTimeString().split(' ')[0] + ' UTC+3';
    }
  }
  updateTime();
  setInterval(updateTime, 1000);

  // CRT Toggle Button
  const crtBtn = document.getElementById('toggle-crt');
  const crtOverlay = document.querySelector('.crt-overlay');
  if (crtBtn && crtOverlay) {
    crtBtn.addEventListener('click', () => {
      crtOverlay.classList.toggle('crt-off');
      const isOff = crtOverlay.classList.contains('crt-off');
      crtBtn.textContent = `[CRT FX: ${isOff ? 'OFF' : 'ON'}]`;
      crtBtn.classList.toggle('active', !isOff);
      playBeep(440, 0.05);
    });
  }

  // Audio Toggle Button
  const soundBtn = document.getElementById('toggle-sound');
  if (soundBtn) {
    soundBtn.addEventListener('click', () => {
      audioEnabled = !audioEnabled;
      soundBtn.textContent = `[AUDIO: ${audioEnabled ? 'ON' : 'OFF'}]`;
      soundBtn.classList.toggle('active', audioEnabled);
      if (audioEnabled) playBeep(880, 0.08);
    });
  }
}

// Retro Web Audio Synthesizer (Zero External Audio Files)
function playBeep(freq = 440, duration = 0.05, type = 'square') {
  if (!audioEnabled) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.05, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    // AudioContext blocked or unsupported
  }
}

/* ==========================================================================
   2. BACKGROUND PARTICLES CANVAS (Floating Ember & Leaves)
   ========================================================================== */
function initBackgroundCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = Array.from({ length: 45 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    size: Math.random() * 3 + 1,
    speedX: Math.random() * 0.6 - 0.3,
    speedY: Math.random() * 0.8 + 0.2,
    opacity: Math.random() * 0.7 + 0.3,
    color: Math.random() > 0.5 ? '#ff6600' : '#ffb700'
  }));

  function render() {
    ctx.clearRect(0, 0, width, height);

    particles.forEach((p) => {
      p.y += p.speedY;
      p.x += Math.sin(p.y * 0.01) * 0.4;

      if (p.y > height) {
        p.y = -10;
        p.x = Math.random() * width;
      }

      ctx.save();
      ctx.globalAlpha = p.opacity;
      ctx.fillStyle = p.color;
      ctx.shadowBlur = 8;
      ctx.shadowColor = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    requestAnimationFrame(render);
  }

  render();
}

/* ==========================================================================
   3. INTERACTIVE TERMINAL CONSOLE & COMMAND PROCESSOR
   ========================================================================== */
function initTerminalConsole() {
  const termBody = document.getElementById('terminal-body');
  const termInput = document.getElementById('command-input');
  if (!termBody || !termInput) return;

  const bootLogs = [
    `[SYSTEM] Initializing QUAKE TERMINAL v3.6.0...`,
    `[SYSTEM] Connecting to host: roman@ort141-kyiv.edu.ua`,
    `[OK] User Authenticated: ROMAN (10th Grade Tech Student)`,
    `[OK] Hobby Detected: Acorn Systematic Collection & Botany`,
    `[SYS] Type '<span class="log-highlight">help</span>' for available terminal commands.`
  ];

  let logIndex = 0;
  function printBootLogs() {
    if (logIndex < bootLogs.length) {
      appendLog(bootLogs[logIndex]);
      logIndex++;
      setTimeout(printBootLogs, 250);
    }
  }
  printBootLogs();

  function appendLog(htmlContent, isInput = false) {
    const line = document.createElement('div');
    line.className = 'log-line';
    const now = new Date().toTimeString().split(' ')[0];

    if (isInput) {
      line.innerHTML = `<span class="log-time">[${now}]</span> <span class="prompt-symbol">roman@ort141:~$</span> <span class="log-highlight">${htmlContent}</span>`;
    } else {
      line.innerHTML = `<span class="log-time">[${now}]</span> ${htmlContent}`;
    }

    termBody.appendChild(line);
    termBody.scrollTop = termBody.scrollHeight;
  }

  termInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const command = termInput.value.trim().toLowerCase();
      if (!command) return;

      appendLog(command, true);
      processCommand(command);
      termInput.value = '';
      playBeep(600, 0.04);
    }
  });

  function processCommand(cmd) {
    switch (cmd) {
      case 'help':
        appendLog(`Available Commands:`);
        appendLog(`  <span class="log-highlight">bio</span>       - Display Roman's student bio`);
        appendLog(`  <span class="log-highlight">school</span>    - School #141 "ORT" Kyiv specifications`);
        appendLog(`  <span class="log-highlight">acorns</span>    - List top acorn collection items`);
        appendLog(`  <span class="log-highlight">game</span>      - Launch acorn clicker focus`);
        appendLog(`  <span class="log-highlight">clear</span>     - Clear terminal buffer`);
        appendLog(`  <span class="log-highlight">matrix</span>    - Display matrix easter egg`);
        appendLog(`  <span class="log-highlight">quake</span>     - Quake easter egg log`);
        appendLog(`  <span class="log-highlight">contact</span>   - Jump to transmission form`);
        break;

      case 'bio':
        appendLog(`<span class="log-success">ROMAN PROFILE:</span> 10th grade student at Kyiv School #141 "ORT". Enthusiastic about web tech, programming, botany, and acorn collecting around Kyiv parks.`);
        break;

      case 'school':
        appendLog(`<span class="log-success">EDUCATIONAL HUB:</span> Educational Complex #141 "ORT" Kyiv. High specialization in STEM, IT, Robotics, and natural sciences.`);
        break;

      case 'acorns':
        appendLog(`<span class="log-success">ACORN DATABASE OVERVIEW:</span> 482+ specimens collected. Species include Quercus robur, Quercus rubra, and rare Golden Acorns.`);
        break;

      case 'game':
        appendLog(`<span class="log-success">[NAVIGATING]</span> Scrolling to Quake Acorn Collector...`);
        document.getElementById('clicker-game')?.scrollIntoView({ behavior: 'smooth' });
        break;

      case 'clear':
        termBody.innerHTML = '';
        break;

      case 'matrix':
        appendLog(`<span class="log-success">01010010 01001111 01001101 01000001 01001110 00100000 01001111 01010010 01010100</span>`);
        appendLog(`<span class="log-highlight">"Wake up, Roman... The oak forest has you."</span>`);
        break;

      case 'quake':
        appendLog(`<span class="log-alert">HUMILIATION!</span> You clicked 1,000 acorns! (Just kidding, keep clicking!)`);
        break;

      case 'contact':
        document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' });
        break;

      default:
        appendLog(`<span class="log-alert">Command not recognized: '${cmd}'. Type 'help' for command list.</span>`);
        break;
    }
  }
}

/* ==========================================================================
   4. NAVIGATION & SMOOTH SCROLLING
   ========================================================================== */
function initNavigation() {
  const toggleBtn = document.getElementById('mobile-toggle');
  const navMenu = document.getElementById('nav-menu');

  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener('click', () => {
      navMenu.classList.toggle('active');
    });
  }

  // Active Nav Link Highlight on Scroll
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach((section) => {
      const sectionTop = section.offsetTop - 120;
      if (window.scrollY >= sectionTop) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}

/* ==========================================================================
   5. ACORN DATABASE & VIRTUAL MUSEUM
   ========================================================================== */
const acornData = [
  {
    id: 'acorn-01',
    name: 'Дуб Звичайний (Quercus robur)',
    latin: 'Quercus robur L.',
    rarity: 'COMMON',
    location: 'Голосіївський Ліс, Київ',
    weight: '4.2 г',
    tannins: 'Високий (8.5%)',
    image: 'assets/hero_avatar.jpg',
    desc: 'Найвідоміший екземпляр дуба в лісах Києва. Має подовжену яйцеподібну форму, глибоку шапочку (плюску) та виражений коричнево-бурштиновий візерунок.',
    tag: 'common'
  },
  {
    id: 'acorn-02',
    name: 'Дуб Червоний (Quercus rubra)',
    latin: 'Quercus rubra L.',
    rarity: 'RARE',
    location: 'Парк КПІ ім. Ігоря Сікорського',
    weight: '5.8 г',
    tannins: 'Середній (6.2%)',
    image: 'assets/acorn_red.jpg',
    desc: 'Північноамериканський вид дуба з короткими широкими жолудями та кулястою плюскою. Вийде знайти восени біля корпусів КПІ.',
    tag: 'rare'
  },
  {
    id: 'acorn-03',
    name: 'Золотий Жолудь Артефакт',
    latin: 'Quercus aurea insignis',
    rarity: 'LEGENDARY',
    location: 'Секретний Дубовий Гай, Феофанія',
    weight: '12.0 г',
    tannins: 'Максимальний (15.0%)',
    image: 'assets/acorn_golden.jpg',
    desc: 'Рідкісний мутований екземпляр з відтінком щирого золота. Символ успіху та головний експонат колекції Романа.',
    tag: 'legendary'
  },
  {
    id: 'acorn-04',
    name: 'Дуб Скельний (Quercus petraea)',
    latin: 'Quercus petraea Liebl.',
    rarity: 'UNCOMMON',
    location: 'Ботанічний сад ім. Фоміна',
    weight: '3.9 г',
    tannins: 'Високий (7.9%)',
    image: 'assets/acorn_common.jpg',
    desc: 'Відрізняється сидячими жолудями без довгих плодоніжок. Зустрічається на підвищеннях та ботанічних ділянках Києва.',
    tag: 'uncommon'
  }
];

function initAcornDatabase() {
  const container = document.getElementById('acorn-grid');
  const filterBtns = document.querySelectorAll('.filter-btn');
  const searchInput = document.getElementById('acorn-search');

  if (!container) return;

  function renderCards(items) {
    container.innerHTML = '';
    items.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'acorn-card';
      const tagClass = item.rarity === 'LEGENDARY' ? 'gold' : item.rarity === 'RARE' ? 'rare' : '';

      card.innerHTML = `
        <div class="acorn-img-wrapper">
          <img src="${item.image}" alt="${item.name}" loading="lazy">
          <span class="rarity-tag ${tagClass}">${item.rarity}</span>
        </div>
        <div class="acorn-title">${item.name}</div>
        <div class="acorn-latin">${item.latin}</div>
        <div class="acorn-specs">
          <div><span>Локація:</span> ${item.location}</div>
          <div><span>Маса:</span> ${item.weight}</div>
        </div>
        <div class="acorn-actions">
          <button class="btn-quake btn-quake-gold btn-block inspect-btn" data-id="${item.id}">
            [INSPECT DATA]
          </button>
        </div>
      `;
      container.appendChild(card);
    });

    // Inspect buttons click listeners
    document.querySelectorAll('.inspect-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-id');
        const specimen = acornData.find((a) => a.id === id);
        if (specimen) openAcornModal(specimen);
      });
    });
  }

  renderCards(acornData);

  // Filter Buttons
  filterBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      filterBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.getAttribute('data-filter');
      if (filter === 'all') {
        renderCards(acornData);
      } else {
        const filtered = acornData.filter((a) => a.tag === filter);
        renderCards(filtered);
      }
      playBeep(520, 0.04);
    });
  });

  // Search Input
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const filtered = acornData.filter(
        (a) => a.name.toLowerCase().includes(query) || a.location.toLowerCase().includes(query)
      );
      renderCards(filtered);
    });
  }
}

// Acorn Modal Inspection
function openAcornModal(specimen) {
  const modal = document.getElementById('acorn-modal');
  const modalBody = document.getElementById('modal-body-content');
  if (!modal || !modalBody) return;

  modalBody.innerHTML = `
    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
      <div style="width: 180px; height: 180px; border: 1px solid var(--amber-primary); flex-shrink: 0;">
        <img src="${specimen.image}" style="width:100%; height:100%; object-fit:cover;">
      </div>
      <div style="flex-grow: 1;">
        <h3 style="color: var(--amber-primary); margin-bottom: 4px;">${specimen.name}</h3>
        <p style="color: var(--green-status); font-style: italic; font-size: 0.85rem; margin-bottom: 12px;">${specimen.latin}</p>
        <div style="font-size: 0.85rem; display: flex; flex-direction: column; gap: 6px; border-top: 1px dashed var(--border-amber); padding-top: 10px;">
          <div><strong>Рідкісність:</strong> <span style="color: var(--gold-accent);">${specimen.rarity}</span></div>
          <div><strong>Локація знахідки:</strong> ${specimen.location}</div>
          <div><strong>Середня маса:</strong> ${specimen.weight}</div>
          <div><strong>Вміст танінів:</strong> ${specimen.tannins}</div>
        </div>
      </div>
    </div>
    <div style="margin-top: 16px; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6; background: rgba(255,102,0,0.05); padding: 12px; border-left: 3px solid var(--amber-primary);">
      ${specimen.desc}
    </div>
  `;

  modal.classList.add('active');
  playBeep(700, 0.06);

  const closeBtn = document.getElementById('close-modal');
  if (closeBtn) {
    closeBtn.onclick = () => modal.classList.remove('active');
  }

  modal.onclick = (e) => {
    if (e.target === modal) modal.classList.remove('active');
  };
}

/* ==========================================================================
   6. QUAKE ACORN COLLECTOR MINI-GAME
   ========================================================================== */
function initClickerGame() {
  const acornTarget = document.getElementById('acorn-click-target');
  const scoreDisplay = document.getElementById('game-score');
  const dpsDisplay = document.getElementById('game-dps');
  const upgradesContainer = document.getElementById('upgrades-container');

  if (!acornTarget || !scoreDisplay) return;

  let state = {
    acorns: 0,
    acornsPerClick: 1,
    dps: 0
  };

  const upgrades = [
    {
      id: 'squirrel',
      name: 'autosquirrel.exe',
      desc: 'Автономна білка збирає 1 жолудь/сек',
      cost: 15,
      dpsAdd: 1,
      count: 0
    },
    {
      id: 'sapling',
      name: 'oak_sapling.sh',
      desc: 'Молодий дубок дає 5 жолудів/сек',
      cost: 100,
      dpsAdd: 5,
      count: 0
    },
    {
      id: 'radar',
      name: 'acorn_radar.bin',
      desc: 'Сенсорний сканер +25 жолудів/сек',
      cost: 500,
      dpsAdd: 25,
      count: 0
    },
    {
      id: 'grove',
      name: 'fortified_grove.sys',
      desc: 'Дубовий гай Феофанії +100 жолудів/сек',
      cost: 2000,
      dpsAdd: 100,
      count: 0
    }
  ];

  // Load state from localStorage
  const saved = localStorage.getItem('roman_acorn_save_v1');
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      state.acorns = parsed.acorns || 0;
      if (parsed.upgrades) {
        parsed.upgrades.forEach((uSaved) => {
          const target = upgrades.find((u) => u.id === uSaved.id);
          if (target) {
            target.count = uSaved.count;
            target.cost = uSaved.cost;
          }
        });
      }
    } catch (e) {
      console.warn('Failed to load save data');
    }
  }

  function calculateDPS() {
    state.dps = upgrades.reduce((acc, u) => acc + u.count * u.dpsAdd, 0);
  }

  function updateUI() {
    calculateDPS();
    scoreDisplay.textContent = Math.floor(state.acorns).toLocaleString();
    if (dpsDisplay) dpsDisplay.textContent = state.dps.toLocaleString();

    renderUpgrades();
  }

  function renderUpgrades() {
    if (!upgradesContainer) return;
    upgradesContainer.innerHTML = '';

    upgrades.forEach((u) => {
      const card = document.createElement('div');
      card.className = 'upgrade-card';

      const canAfford = state.acorns >= u.cost;
      card.innerHTML = `
        <div class="upgrade-info">
          <div class="upgrade-name">${u.name} <span style="color: var(--green-status)">[x${u.count}]</span></div>
          <div class="upgrade-desc">${u.desc}</div>
          <div class="upgrade-cost">Ціна: ${u.cost} жолудів</div>
        </div>
        <button class="btn-quake buy-upgrade-btn upgrade-buy-btn" data-id="${u.id}" ${canAfford ? '' : 'disabled'}>
          [BUY]
        </button>
      `;
      upgradesContainer.appendChild(card);
    });

    document.querySelectorAll('.buy-upgrade-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-id');
        const up = upgrades.find((u) => u.id === id);
        if (up && state.acorns >= up.cost) {
          state.acorns -= up.cost;
          up.count++;
          up.cost = Math.floor(up.cost * 1.35);
          saveGame();
          updateUI();
          playBeep(800, 0.08, 'sawtooth');
          showToast(`Куплено: ${up.name}!`);
        }
      });
    });
  }

  // Click Target Interaction
  acornTarget.addEventListener('click', (e) => {
    state.acorns += state.acornsPerClick;

    // Floating Popup Text
    createClickPopup(e.clientX, e.clientY, `+${state.acornsPerClick}`);

    playBeep(300 + Math.random() * 200, 0.04);
    updateUI();
    saveGame();
  });

  function createClickPopup(x, y, text) {
    const popup = document.createElement('div');
    popup.className = 'click-popup';
    popup.textContent = text;
    popup.style.left = `${x - 15}px`;
    popup.style.top = `${y - 25}px`;
    document.body.appendChild(popup);

    setTimeout(() => popup.remove(), 800);
  }

  // Auto DPS Game Loop
  setInterval(() => {
    if (state.dps > 0) {
      state.acorns += state.dps / 10;
      updateUI();
    }
  }, 100);

  // Auto Save every 10 sec
  setInterval(saveGame, 10000);

  function saveGame() {
    const toSave = {
      acorns: state.acorns,
      upgrades: upgrades.map((u) => ({ id: u.id, count: u.count, cost: u.cost }))
    };
    localStorage.setItem('roman_acorn_save_v1', JSON.stringify(toSave));
  }

  updateUI();
}

/* ==========================================================================
   7. TRANSMISSION FORM HANDLER
   ========================================================================== */
function initTransmissionForm() {
  const form = document.getElementById('contact-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const nameInput = document.getElementById('input-name');
    const msgInput = document.getElementById('input-message');

    if (!nameInput.value.trim() || !msgInput.value.trim()) {
      showToast('Помилка: Заповніть усі поля форми!');
      playBeep(200, 0.1, 'sawtooth');
      return;
    }

    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      const origText = btn.textContent;
      btn.textContent = '[TRANSMITTING...]';
      btn.disabled = true;

      setTimeout(() => {
        showToast('Повідомлення успішно відправлено на консоль Романа!');
        playBeep(900, 0.15);
        btn.textContent = origText;
        btn.disabled = false;
        form.reset();
      }, 1200);
    }
  });
}

// Toast Notification Helper
function showToast(message) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `<span style="color: var(--green-status); font-weight: bold;">[SYS_MSG]:</span> ${message}`;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
