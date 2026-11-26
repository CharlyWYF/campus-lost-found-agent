/* 寻物有迹 · 自动演示（纯静态，不调用 API） */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const PARAMS = new URLSearchParams(location.search);
const RAW_SPEED = Number(PARAMS.get("speed") || 1);
const SPEED = Number.isFinite(RAW_SPEED) ? Math.max(0.5, Math.min(4, RAW_SPEED)) : 1;
const HOLD_TICKET = PARAMS.has("holdTicket");
const RAW_VOICE_RATE = Number(PARAMS.get("voiceSpeed") || 1.35);
const VOICE_RATE = Number.isFinite(RAW_VOICE_RATE) ? Math.max(0.75, Math.min(1.6, RAW_VOICE_RATE)) : 1.35;

const els = {
  intro: $("#intro"),
  outro: $("#outro"),
  student: $("#app-student"),
  admin: $("#app-admin"),
  messages: $("#demo-messages"),
  slots: $("#demo-slots"),
  candidates: $("#demo-candidates"),
  trace: $("#demo-trace"),
  candidateCount: $("#candidate-count"),
  adminMessages: $("#admin-messages"),
  adminTable: $("#admin-table"),
  adminPending: $("#admin-pending"),
  auditLine: $("#audit-line"),
  captionStep: $("#caption-step"),
  captionTitle: $("#caption-title"),
  captionText: $("#caption-text"),
  narration: $("#narration"),
  narrationText: $("#narration-text"),
  startBtn: $("#start-btn"),
  silentBtn: $("#silent-btn"),
  progress: $("#progress-fill"),
  sceneLabel: $("#scene-label"),
  voiceBtn: $("#voice-btn"),
  pauseBtn: $("#pause-btn"),
  replayBtn: $("#replay-btn"),
  againBtn: $("#again-btn"),
  confirmBtn: $("#confirm-btn"),
  statTotal: $("#stat-total"),
  statOpen: $("#stat-open"),
  statClaimed: $("#stat-claimed"),
  statReturned: $("#stat-returned"),
};

let paused = false;
let finished = false;
let demoStarted = false;
let step = 1;
let voiceOn = true;

const voice = new Audio();
voice.preload = "auto";
voice.volume = 0.92;
voice.playbackRate = VOICE_RATE;
voice.preservesPitch = true;

const CAPTIONS = {
  1: {
    step: "01 / 拾得登记",
    title: "捡到的人，先登记",
    text: "只有主人知道的细节，被单独保存，不公开。",
    scene: "拾得登记",
    progress: 8,
  },
  2: {
    step: "02 / 失主认领",
    title: "失主回来，先比对再核验",
    text: "不是看到“像”就给，先问只有主人知道的细节。",
    scene: "失主认领",
    progress: 40,
  },
  3: {
    step: "03 / 管理台交接",
    title: "老师这边，查、看、改、销",
    text: "写操作先预览，确认后才落库，全程审计。",
    scene: "管理台交接",
    progress: 74,
  },
};

/* ------------------------------------------------------------------ timing */
async function wait(ms) {
  let remain = ms / SPEED;
  while (remain > 0) {
    const slice = Math.min(40, remain);
    await new Promise((resolve) => setTimeout(resolve, slice));
    if (!paused) remain -= slice;
  }
}

async function typeText(el, text, speed = 16) {
  el.textContent = "";
  for (const char of text) {
    while (paused) await new Promise((resolve) => setTimeout(resolve, 80));
    el.textContent += char;
    await wait(speed);
  }
}

function scrollBottom(el) {
  requestAnimationFrame(() => {
    el.scrollTop = el.scrollHeight;
  });
}

/* ------------------------------------------------------------------ chrome */
function setStep(n) {
  step = n;
  const cap = CAPTIONS[n];
  els.captionStep.textContent = cap.step;
  els.captionTitle.textContent = cap.title;
  els.captionText.textContent = cap.text;
  els.sceneLabel.textContent = cap.scene;
  els.progress.style.width = `${cap.progress}%`;
  $$(".step").forEach((item) => item.classList.toggle("active", Number(item.dataset.step) === n));
  document.querySelector(".eyebrow").textContent = `AUTO DEMO / 0${n}`;
}

function showApp(name) {
  els.student.classList.toggle("hidden", name !== "student");
  els.admin.classList.toggle("hidden", name !== "admin");
  const tabs = $$(".app-tabs span");
  if (tabs.length >= 2) {
    tabs[0].classList.toggle("on", name === "student");
    tabs[1].classList.toggle("on", name === "admin");
  }
}

function clearMessages(container) {
  container.innerHTML = "";
}

/* ------------------------------------------------------------------ messages */
function makeMessage(container, role, text) {
  const empty = container.querySelector(".empty-hint");
  if (empty) empty.remove();
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const avatar = role === "user" ? "你" : "寻";
  const name = role === "user" ? "你" : "小寻";
  wrapper.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-name">${name}</div>
      <div class="msg-bubble"></div>
      <div class="msg-cards"></div>
    </div>`;
  const bubble = $(".msg-bubble", wrapper);
  if (text) bubble.textContent = text;
  container.appendChild(wrapper);
  scrollBottom(container);
  return { wrapper, bubble, cards: $(".msg-cards", wrapper) };
}

async function userSay(text, speed = 10) {
  const { bubble } = makeMessage(els.messages, "user", "");
  await typeText(bubble, text, speed);
  scrollBottom(els.messages);
  await wait(750);
}

async function assistantSay(text, options = {}) {
  const { bubble, cards } = makeMessage(els.messages, "assistant", "");
  bubble.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
  scrollBottom(els.messages);
  await wait(options.thinking || 1100);
  bubble.innerHTML = "";
  await typeText(bubble, text, options.speed || 18);
  if (options.card) {
    cards.innerHTML = options.card;
    await wait(90);
    if (HOLD_TICKET && options.card.includes("mini-ticket")) {
      paused = true;
      els.pauseBtn.textContent = "继续";
      if (!voice.paused) voice.pause();
    }
  }
  scrollBottom(els.messages);
  await wait(options.after || 700);
}

function addAdminMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const avatar = role === "user" ? "你" : "寻";
  wrapper.innerHTML = `<div class="msg-avatar">${avatar}</div><div class="msg-body"><div class="msg-bubble"></div></div>`;
  const bubble = $(".msg-bubble", wrapper);
  bubble.textContent = text;
  els.adminMessages.appendChild(wrapper);
  scrollBottom(els.adminMessages);
}

/* ------------------------------------------------------------------ workspace */
function renderSlots(items) {
  els.slots.innerHTML = items
    .map(([label, value]) => `<span class="slot"><b>${label}</b>${value}</span>`)
    .join("");
}

function renderCandidates(items) {
  els.candidateCount.textContent = String(items.length);
  els.candidates.innerHTML = items.length
    ? items.map((item, index) => `
      <div class="cand ${index === 0 ? "best" : ""}">
        <div class="cand-top"><span class="cand-title">${item.title}</span><span class="cand-score">${item.score}</span></div>
        <div class="cand-sub">${item.sub}</div>
        <div class="cand-bar"><span style="width:${Math.round(item.value * 100)}%"></span></div>
        <div class="cand-tags">${item.tags.map((tag) => `<span>${tag}</span>`).join("")}</div>
      </div>`).join("")
    : `<span class="slot-placeholder">还没有开始匹配。</span>`;
}

function addTrace(tool, detail) {
  const item = document.createElement("div");
  item.className = "trace-item";
  item.innerHTML = `<b>${tool}</b><span>${detail}</span>`;
  if (els.trace.querySelector(".slot-placeholder")) els.trace.innerHTML = "";
  els.trace.appendChild(item);
  scrollBottom(els.trace);
}

function setStatus(caseId, status, label) {
  const row = els.adminTable.querySelector(`.tr[data-id="${caseId}"]`);
  if (!row) return;
  const cell = row.querySelector("[data-status]");
  if (cell) {
    cell.className = `st ${status}`;
    cell.textContent = label;
  }
  row.classList.add("highlight");
}

/* ------------------------------------------------------------------ narration */
function setNarration(text) {
  els.narrationText.textContent = text;
  els.narration.classList.remove("flash");
  void els.narration.offsetWidth;
  els.narration.classList.add("flash");
}

function stopVoice() {
  if (!voice.paused) voice.pause();
  voice.currentTime = 0;
}

function narrate(name, text, minMs = 3000) {
  setNarration(text);
  if (!voiceOn) return wait(minMs);
  stopVoice();
  voice.src = `narration/${name}.mp3`;
  voice.playbackRate = VOICE_RATE;
  voice.preservesPitch = true;
  return new Promise((resolve) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      voice.onended = null;
      voice.onerror = null;
      resolve();
    };
    voice.onended = done;
    voice.onerror = done;
    const playing = voice.play();
    if (playing) playing.catch(done);
    setTimeout(done, 24000);
  });
}

/* ------------------------------------------------------------------ scenes */
async function sceneOne() {
  setStep(1);
  renderSlots([["角色", "拾得者"]]);
  const narration = narrate("scene1", "有同学在图书馆捡到一个耳机盒，交到服务台。小寻先记下公开信息，再问只有主人知道的细节。", 12000);
  await wait(700);

  await userSay("我在图书馆三楼靠窗座位捡到一个黑色耳机盒，上面挂着黄色小鸭子挂件，已经交到一楼服务台了。");
  addTrace("set_role", "身份：拾得者");
  renderSlots([
    ["类别", "耳机盒"],
    ["颜色", "黑色"],
    ["地点", "图书馆三楼"],
    ["标记", "黄色小鸭子挂件"],
  ]);
  await wait(500);

  await assistantSay("好，先记一下。它有没有什么只有主人才知道的细节？比如盒内的贴纸、刻字，或者耳机本身的颜色。");
  addTrace("update_draft", "记录公开信息，继续追问核验点");
  await wait(500);

  await userSay("盒盖内侧贴了张蓝色皮卡丘贴纸，耳机是白色的，充电口旁边有道小划痕。");
  addTrace("update_draft", "私密细节单独保存，不写进公开描述");
  renderSlots([
    ["类别", "耳机盒"],
    ["颜色", "黑色"],
    ["地点", "图书馆三楼"],
    ["标记", "黄色小鸭子挂件"],
    ["阶段", "已登记"],
  ]);
  await wait(600);

  await assistantSay("记下了。这个细节只用来核对，不会公开。案件编号 F-20261126-001。", {
    card: `<div class="mini-case"><b>拾得案件 · F-20261126-001</b><span>已登记</span></div>`,
    after: 1400,
  });
  await narration;
}

async function sceneTwo() {
  setStep(2);
  clearMessages(els.messages);
  els.candidates.innerHTML = `<span class="slot-placeholder">还没有开始匹配。</span>`;
  els.trace.innerHTML = `<span class="slot-placeholder">工具调用会显示在这里。</span>`;
  els.candidateCount.textContent = "0";
  renderSlots([["角色", "失主"]]);
  const narration = narrate("scene2", "失主回来，只记得大概。小寻先匹配，再一条条核对。三项都对上，才给认领码。", 13000);
  await wait(800);

  await userSay("我昨天下午在图书馆三楼丢了一个黑色 AirPods 盒子，上面挂着黄色小鸭子挂件。");
  addTrace("set_role", "身份：失主");
  renderSlots([
    ["类别", "耳机盒"],
    ["颜色", "黑色"],
    ["地点", "图书馆三楼"],
    ["时间", "昨天下午"],
  ]);
  await wait(600);

  await assistantSay("我找到一条挺像的记录：昨天下午图书馆三楼靠窗，有人交来一个黑色耳机盒。先核对一下：盒盖内侧有什么贴纸或图案？", {
    card: `<div class="mini-case"><b>匹配到 1 条记录 · 0.95</b><span>进入核验</span></div>`,
    after: 700,
  });
  addTrace("search_found_cases", "检索到 1 条高度吻合的记录");
  renderCandidates([
    { title: "黑色 AirPods 耳机盒", sub: "图书馆三楼靠窗 · 昨天 16:30", score: "0.95", value: 0.95, tags: ["同类", "黑色", "图书馆三楼", "小鸭子挂件"] },
    { title: "黑色有线耳机", sub: "图书馆三楼自习区 · 昨天 15:10", score: "0.30", value: 0.30, tags: ["同类", "黑色", "地点接近"] },
    { title: "白色 AirPods 充电盒", sub: "教学楼 204 · 昨天 10:20", score: "0.22", value: 0.22, tags: ["同类", "时间接近"] },
  ]);
  await wait(800);

  await userSay("蓝色皮卡丘贴纸。");
  await assistantSay("对上了。再确认：耳机本体是什么颜色？", { thinking: 700, speed: 13 });
  await wait(450);

  await userSay("白色。");
  await assistantSay("最后一问：充电口附近有划痕吗？", { thinking: 700, speed: 13 });
  await wait(450);

  await userSay("有，一道小划痕。");
  renderSlots([
    ["类别", "耳机盒"],
    ["颜色", "黑色"],
    ["地点", "图书馆三楼"],
    ["时间", "昨天下午"],
    ["认领码", "LF-CUYC"],
  ]);
  addTrace("submit_verification_answer", "三项核验通过");
  await assistantSay("核对通过。认领码 LF-CUYC，请到图书馆一楼服务台领取。", {
    card: `
      <div class="mini-ticket">
        <div class="tk-label"><span>认领凭证</span><b>核验通过</b></div>
        <div class="tk-code">LF-CUYC</div>
        <div class="tk-meta">领取地点：图书馆一楼服务台</div>
      </div>`,
    after: 2800,
  });
  addTrace("finalize_claim", "生成认领码 LF-CUYC");
  els.captionStep.textContent = "02 / 核验通过";
  els.captionTitle.textContent = "认领码 LF-CUYC";
  els.captionText.textContent = "三项细节都对上，才给凭证。";
  els.progress.style.width = "60%";
  await narration;
}

async function sceneThree() {
  setStep(3);
  showApp("admin");
  const narration = narrate("scene3", "管理员在后台查记录、看核验细节，先预览再确认。核销后，状态变成已归还，审计也留了下来。", 14000);
  await wait(1100);

  addAdminMessage("user", "查一下这条认领记录。");
  await wait(700);
  addAdminMessage("assistant", "F-20261126-001，黑色 AirPods 耳机盒，认领码 LF-CUYC，三项核验都通过。");
  setStatus("F-20261126-001", "claimed", "已认领");
  await wait(1000);

  addAdminMessage("user", "核销认领码 LF-CUYC。");
  await wait(700);
  addAdminMessage("assistant", "先确认一下：核销后，案件会变成已归还。");
  els.adminPending.classList.remove("hidden");
  els.confirmBtn.classList.add("pulse");
  await wait(3500);

  els.confirmBtn.classList.remove("pulse");
  els.confirmBtn.textContent = "已确认";
  await wait(700);
  els.adminPending.classList.add("hidden");
  setStatus("F-20261126-001", "returned", "已归还");
  els.statClaimed.textContent = "0";
  els.statReturned.textContent = "1";
  els.auditLine.classList.remove("hidden");
  addAdminMessage("assistant", "核销完成，案件已归还，审计已记录。");
  els.captionStep.textContent = "03 / 交接完成";
  els.captionTitle.textContent = "已归还，审计已记录";
  els.captionText.textContent = "状态变了，操作也留下了痕迹。";
  els.progress.style.width = "94%";
  await wait(2600);
  await narration;
}

async function finish() {
  els.progress.style.width = "100%";
  els.sceneLabel.textContent = "演示完成";
  const narration = narrate("outro", "它不是一个更聪明的搜索框。它把信息问全，把两边对上，还要对东西交给谁负责。", 9000);
  await wait(600);
  els.outro.classList.remove("hidden");
  await narration;
  if (PARAMS.has("loop")) {
    await wait(5000);
    location.reload();
  }
}

/* ------------------------------------------------------------------ controls */
async function run() {
  await wait(900);
  await sceneOne();
  await wait(3200);
  await sceneTwo();
  await wait(4000);
  await sceneThree();
  await wait(5000);
  await finish();
  finished = true;
}

els.pauseBtn.addEventListener("click", () => {
  paused = !paused;
  els.pauseBtn.textContent = paused ? "继续" : "暂停";
  if (paused) {
    if (!voice.paused) voice.pause();
  } else if (voiceOn && voice.src && !voice.ended) {
    voice.play().catch(() => {});
  }
});

els.voiceBtn.addEventListener("click", () => {
  voiceOn = !voiceOn;
  els.voiceBtn.textContent = voiceOn ? "旁白：开" : "旁白：关";
  if (!voiceOn) {
    if (!voice.paused) voice.pause();
  } else if (voice.src && !voice.ended) {
    voice.play().catch(() => {});
  }
});

els.replayBtn.addEventListener("click", () => location.reload());
els.againBtn.addEventListener("click", () => location.reload());

window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    els.pauseBtn.click();
  }
});

async function startDemo(withVoice) {
  if (demoStarted) return;
  demoStarted = true;
  voiceOn = withVoice;
  els.pauseBtn.textContent = "暂停";
  els.voiceBtn.textContent = voiceOn ? "旁白：开" : "旁白：关";
  const introNarration = narrate("intro", "这是一个校园里很常见的下午。", 3000);
  els.intro.classList.add("hide");
  await wait(700);
  await introNarration;
  await run();
}

els.startBtn.addEventListener("click", () => startDemo(true));
els.silentBtn.addEventListener("click", () => startDemo(false));

if (PARAMS.has("auto")) {
  startDemo(false);
}

window.__demo = {
  get step() { return step; },
  get paused() { return paused; },
  get finished() { return finished; },
  get voiceOn() { return voiceOn; },
  get voice() {
    return {
      src: voice.src,
      paused: voice.paused,
      time: voice.currentTime,
      duration: voice.duration,
      rate: voice.playbackRate,
      pitch: voice.preservesPitch,
      error: voice.error ? voice.error.code : null,
    };
  },
};
