// ========== State ==========
let currentSessionId = null;
let isLoading = false;
let isTyping = false;
let isWebSearchMode = false;

const API_BASE = '';

// ========== DOM cache ==========
const $chatArea = $('#chat-output');
const $userInput = $('#user-query');
const $sendBtn = $('#send-button');
const $welcomeScreen = $('#welcome-screen');
const $refsPanel = $('#relevant-documents-container');
const $sessionList = $('#chat-sessions');
const $modeHint = $('#mode-hint');
const $webToggle = $('#toggle-search-web');
const $sidebar = $('#sidebar');

// ========== Utilities ==========
function escapeHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatBotText(s) {
    if (!s) return '';
    let out = escapeHtml(s);
    out = out.replace(/\\n/g, '\n');
    out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    out = out.replace(/\n/g, '<br>');
    return out;
}

function scrollToBottom(force = false) {
    const el = $chatArea[0];
    if (!el) return;
    if (!force) {
        const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        if (distanceFromBottom > 120) return;
    }
    el.scrollTop = el.scrollHeight;
}

function showToast(text, type = '') {
    const $t = $(`<div class="toast ${type}">${escapeHtml(text)}</div>`);
    $('#toast-container').append($t);
    setTimeout(() => $t.fadeOut(300, () => $t.remove()), 3000);
}

function hideWelcome() { $welcomeScreen.hide(); }

function showWelcome() {
    $chatArea.find('.msg-group, .msg-user, .msg-bot').remove();
    $welcomeScreen.show();
    $refsPanel.empty();
}

function formatMs(ms) {
    if (ms == null) return '';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

// ========== Session management ==========
function startNewSession(silent = false) {
    $.ajax({
        url: `${API_BASE}/api/session/start-session`,
        type: 'POST',
        success: (res) => {
            currentSessionId = res.session_id;
            localStorage.setItem('session_id', currentSessionId);
            showWelcome();
            $sessionList.find('.chat-session').removeClass('active');
            $userInput.prop('disabled', false).focus();
            if (!silent) showToast('Đã tạo cuộc trò chuyện mới', 'success');
        },
        error: () => showToast('Không thể tạo phiên mới', 'error')
    });
}

function loadSessions() {
    $.ajax({
        url: `${API_BASE}/api/session/get-sessions`,
        type: 'GET',
        success: (res) => {
            $sessionList.empty();
            (res.sessions || []).forEach((s) => {
                const title = (s.first_message || 'Cuộc trò chuyện').slice(0, 40);
                const $item = $(`
                    <div class="chat-session" data-session-id="${s.id}">
                        <div class="chat-session-content">${escapeHtml(title)}</div>
                        <button class="session-delete-btn" title="Xóa">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6"/></svg>
                        </button>
                    </div>
                `);
                if (s.id === currentSessionId) $item.addClass('active');
                $item.on('click', (e) => {
                    if ($(e.target).closest('.session-delete-btn').length) return;
                    loadHistory(s.id);
                });
                $item.find('.session-delete-btn').on('click', (e) => {
                    e.stopPropagation();
                    deleteSession(s.id);
                });
                $sessionList.append($item);
            });
        }
    });
}

function deleteSession(sessionId) {
    if (!confirm('Bạn có chắc muốn xóa cuộc trò chuyện này?')) return;
    $.ajax({
        url: `${API_BASE}/api/session/delete-session/${sessionId}`,
        type: 'DELETE',
        success: () => {
            showToast('Đã xóa cuộc trò chuyện', 'success');
            if (sessionId === currentSessionId) startNewSession(true);
            loadSessions();
        },
        error: () => showToast('Không thể xóa', 'error')
    });
}

function loadHistory(sessionId) {
    currentSessionId = sessionId;
    localStorage.setItem('session_id', sessionId);
    $sessionList.find('.chat-session').removeClass('active');
    $sessionList.find(`.chat-session[data-session-id="${sessionId}"]`).addClass('active');
    $.ajax({
        url: `${API_BASE}/api/session/get-chat-history/${sessionId}`,
        type: 'GET',
        success: (res) => {
            $chatArea.find('.msg-group, .msg-user, .msg-bot').remove();
            $welcomeScreen.hide();
            $refsPanel.empty();
            const $group = $('<div class="msg-group"></div>').appendTo($chatArea);
            (res.chat_history || []).forEach((m) => {
                if (m.sender === 'user') {
                    $group.append(renderUserMessage(m.message));
                } else {
                    $group.append(renderBotMessage(m.message, null, m.id));
                }
            });
            scrollToBottom();
        }
    });
}

// ========== Message rendering ==========
function renderUserMessage(text) {
    return $(`
        <div class="msg-user">
            <div class="avatar avatar-user">👤</div>
            <div class="bubble">${escapeHtml(text)}</div>
        </div>
    `);
}

function renderBotMessage(text, references, messageId) {
    const $msg = $(`
        <div class="msg-bot">
            <div class="avatar avatar-bot">⚖️</div>
            <div class="bubble"></div>
        </div>
    `);
    const $bubble = $msg.find('.bubble');
    $bubble.html(formatBotText(text));

    if (references && Array.isArray(references) && references.length > 0) {
        appendReferences($bubble, references);
    } else if (messageId) {
        $.ajax({
            url: `${API_BASE}/api/session/get-message-references/${messageId}`,
            type: 'GET',
            success: (res) => {
                if (res.references && res.references.length > 0) {
                    appendReferences($bubble, res.references);
                }
            }
        });
    }
    return $msg;
}

function appendReferences($bubble, refs) {
    const $refs = $(`
        <div class="references">
            <div class="ref-toggle">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                <span>Tài liệu tham khảo (${refs.length})</span>
            </div>
            <div class="ref-list"></div>
        </div>
    `);
    const $list = $refs.find('.ref-list');
    refs.forEach((r) => {
        const isUrl = typeof r === 'string' && /^https?:\/\//.test(r.trim());
        if (isUrl) {
            $list.append(`<div class="ref-item"><a href="${escapeHtml(r)}" target="_blank" rel="noopener">${escapeHtml(r)}</a></div>`);
        } else {
            $list.append(`<div class="ref-item">${escapeHtml(r)}</div>`);
        }
    });
    $refs.find('.ref-toggle').on('click', () => $list.toggleClass('open'));
    $bubble.append($refs);
}

// ========== Agent thinking panel ==========
// Visual metadata per agent. The orchestrator emits events tagged with
// `agent` (planner, retriever, extractor, grader, web_researcher, generator,
// fact_checker, orchestrator). Each gets its own colour + icon.
const AGENT_META = {
    planner:        { icon: '🧭', label: 'Planner',       klass: 'agent-planner' },
    retriever:      { icon: '🔎', label: 'Retriever',     klass: 'agent-retriever' },
    extractor:      { icon: '✂️', label: 'Extractor',     klass: 'agent-extractor' },
    grader:         { icon: '🎯', label: 'Grader',        klass: 'agent-grader' },
    web_researcher: { icon: '🌐', label: 'Web Research',  klass: 'agent-web' },
    generator:      { icon: '✍️', label: 'Generator',     klass: 'agent-generator' },
    fact_checker:   { icon: '🛡️', label: 'Fact Checker',  klass: 'agent-checker' },
    orchestrator:   { icon: '⚙️', label: 'Orchestrator',  klass: 'agent-orchestrator' },
};

function createThinkingPanel() {
    const $panel = $(`
        <div class="thinking">
            <div class="thinking-header">
                <div class="thinking-header-left">
                    <div class="thinking-spinner"></div>
                    <span class="thinking-title">Agent đang suy luận...</span>
                </div>
                <div class="thinking-header-right">
                    <span class="thinking-elapsed">00:00</span>
                    <svg class="thinking-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
                </div>
            </div>
            <div class="thinking-steps"></div>
        </div>
    `);
    $panel.find('.thinking-header').on('click', () => $panel.toggleClass('collapsed'));

    const startTime = Date.now();
    const timer = setInterval(() => {
        const el = Math.floor((Date.now() - startTime) / 1000);
        const mm = String(Math.floor(el / 60)).padStart(2, '0');
        const ss = String(el % 60).padStart(2, '0');
        $panel.find('.thinking-elapsed').text(`${mm}:${ss}`);
    }, 1000);

    return {
        $el: $panel,
        addStep(event) { renderAgentEvent($panel, event); scrollToBottom(); },
        finish(label) {
            clearInterval(timer);
            $panel.find('.thinking-spinner').replaceWith(`
                <svg class="thinking-done-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            `);
            $panel.find('.thinking-title').text(label || 'Hoàn tất');
            setTimeout(() => $panel.addClass('collapsed'), 1400);
        },
        fail(label) {
            clearInterval(timer);
            $panel.find('.thinking-spinner').replaceWith(`<span class="thinking-fail-icon">✕</span>`);
            $panel.find('.thinking-title').text(label || 'Lỗi');
        },
    };
}

function renderAgentEvent($panel, ev) {
    const agentMeta = AGENT_META[ev.agent] || { icon: '•', label: ev.agent || 'agent', klass: '' };
    const $steps = $panel.find('.thinking-steps');

    // Build the step body based on event type
    const body = buildEventBody(ev);
    if (!body) return;

    const elapsed = ev.elapsed_ms != null ? `<span class="step-elapsed">${formatMs(ev.elapsed_ms)}</span>` : '';
    const $step = $(`
        <div class="agent-step ${agentMeta.klass}">
            <div class="agent-step-head">
                <span class="agent-icon">${agentMeta.icon}</span>
                <span class="agent-name">${agentMeta.label}</span>
                <span class="agent-action">${body.title}</span>
                ${elapsed}
            </div>
            ${body.content ? `<div class="agent-step-body">${body.content}</div>` : ''}
        </div>
    `);
    $steps.append($step);
}

function buildEventBody(ev) {
    // Each handler returns { title, content }. Content is trusted HTML and
    // must escape any user/model-sourced strings before embedding.
    switch (ev.type) {
        case 'plan': {
            const intentLabel = {
                chitchat: 'Chào hỏi',
                legal: 'Câu hỏi pháp luật',
                realtime: 'Thông tin thời gian thực',
                out_of_scope: 'Ngoài phạm vi',
            }[ev.intent] || ev.intent;
            const toolLabel = {
                template: 'Trả lời mẫu',
                rag: 'RAG nội bộ',
                web: 'Web Search',
                reject: 'Từ chối lịch sự',
                ask_user: 'Hỏi lại người dùng',
            }[ev.tool] || ev.tool;

            let rows = `<div class="kv"><span class="k">Ý định</span><span class="v tag tag-intent">${escapeHtml(intentLabel)}</span></div>`;
            rows += `<div class="kv"><span class="k">Tool chọn</span><span class="v tag tag-tool">${escapeHtml(toolLabel)}</span></div>`;
            if (ev.rewritten_query && ev.rewritten_query !== '') {
                rows += `<div class="kv"><span class="k">Query đã chuẩn hoá</span><span class="v">${escapeHtml(ev.rewritten_query)}</span></div>`;
            }
            if (ev.search_query) {
                rows += `<div class="kv"><span class="k">Query tìm web</span><span class="v">${escapeHtml(ev.search_query)}</span></div>`;
            }
            if (ev.entities && Object.keys(ev.entities).length > 0) {
                const ent = Object.entries(ev.entities)
                    .map(([k, v]) => `<span class="chip">${escapeHtml(k)}: ${escapeHtml(v)}</span>`).join('');
                rows += `<div class="kv"><span class="k">Metadata filter</span><span class="v chips">${ent}</span></div>`;
            }
            if (ev.reason) rows += `<div class="reason">${escapeHtml(ev.reason)}</div>`;
            return { title: 'Lập kế hoạch', content: rows };
        }
        case 'retrieve_start':
            return {
                title: `Truy xuất Qdrant ${ev.iteration > 1 ? `(vòng ${ev.iteration})` : ''}`,
                content: `<div class="reason">Tìm với query: <em>${escapeHtml(ev.query)}</em>${
                    ev.entities ? ' · có filter metadata' : ''
                }</div>`,
            };
        case 'retrieve_done': {
            let body = null;
            if (ev.docs && ev.docs.length > 0) {
                const chips = ev.docs
                    .map(d => `<span class="chip chip-doc" title="${escapeHtml(d)}">${escapeHtml(d)}</span>`)
                    .join('');
                body = `<div class="chips">${chips}</div>`;
            }
            return {
                title: `Nhận được ${ev.n_docs} văn bản`,
                content: body,
            };
        }
        case 'extract':
            return {
                title: `Trích đoạn bằng BERT QA (${ev.n_snippets} snippet)`,
                content: null,
            };
        case 'grade': {
            const decisionLabel = {
                sufficient: 'Đủ thông tin',
                partial: 'Còn thiếu',
                insufficient: 'Không đủ',
            }[ev.decision] || ev.decision;
            const scorePct = Math.round((ev.relevance_score || 0) * 100);
            let rows = `<div class="kv"><span class="k">Điểm liên quan</span><span class="v"><div class="score-bar"><div class="score-fill" style="width:${scorePct}%"></div></div><span class="score-text">${scorePct}%</span></span></div>`;
            rows += `<div class="kv"><span class="k">Đánh giá</span><span class="v tag tag-${ev.decision}">${escapeHtml(decisionLabel)}</span></div>`;
            if (ev.missing_aspects && ev.missing_aspects.length > 0) {
                const items = ev.missing_aspects.map(a => `<span class="chip chip-warn">${escapeHtml(a)}</span>`).join('');
                rows += `<div class="kv"><span class="k">Còn thiếu</span><span class="v chips">${items}</span></div>`;
            }
            if (ev.reason) rows += `<div class="reason">${escapeHtml(ev.reason)}</div>`;
            return { title: 'Đánh giá retrieval', content: rows };
        }
        case 'iterate':
            return {
                title: `Điều chỉnh chiến lược${ev.next_iteration ? ` → vòng ${ev.next_iteration}` : ''}`,
                content: `
                    ${ev.next_query ? `<div class="kv"><span class="k">Query mới</span><span class="v">${escapeHtml(ev.next_query)}</span></div>` : ''}
                    ${ev.reason ? `<div class="reason">${escapeHtml(ev.reason)}</div>` : ''}
                `,
            };
        case 'web_start':
            return {
                title: 'Tìm trên Tavily',
                content: `<div class="reason">Query: <em>${escapeHtml(ev.query)}</em></div>`,
            };
        case 'web_done': {
            const links = (ev.links || []).map(u => `<a href="${escapeHtml(u)}" target="_blank" rel="noopener" class="chip chip-link">${escapeHtml(u)}</a>`).join('');
            return {
                title: `Nhận ${ev.n_links} link`,
                content: links ? `<div class="chips">${links}</div>` : null,
            };
        }
        case 'generate_start': {
            const modeLabel = {
                rag: 'tổng hợp từ văn bản luật',
                web_synthesize: 'tổng hợp từ web',
                template: 'trả lời mẫu',
                reject: 'từ chối lịch sự',
            }[ev.mode] || ev.mode;
            return {
                title: `Tổng hợp câu trả lời (${modeLabel})`,
                content: ev.reason ? `<div class="reason">${escapeHtml(ev.reason)}</div>` : null,
            };
        }
        case 'generate_done':
            return {
                title: ev.regenerated ? 'Tổng hợp lại' : `Tổng hợp xong${ev.n_citations != null ? ` (${ev.n_citations} trích dẫn)` : ''}`,
                content: null,
            };
        case 'check': {
            const badge = ev.is_grounded
                ? `<span class="tag tag-ok">Grounded</span>`
                : `<span class="tag tag-warn">Phát hiện claim không có trong nguồn</span>`;
            let rows = `<div class="kv"><span class="k">Kết quả</span><span class="v">${badge}</span></div>`;
            rows += `<div class="kv"><span class="k">Độ tự tin</span><span class="v">${Math.round((ev.confidence || 0) * 100)}%</span></div>`;
            if (ev.unsupported_claims && ev.unsupported_claims.length > 0) {
                const items = ev.unsupported_claims.map(c => `<li>${escapeHtml(c)}</li>`).join('');
                rows += `<div class="kv col"><span class="k">Claim không có trong nguồn</span><ul class="claim-list">${items}</ul></div>`;
            }
            if (ev.reason) rows += `<div class="reason">${escapeHtml(ev.reason)}</div>`;
            return { title: 'Kiểm tra hallucination', content: rows };
        }
        case 'regenerate':
            return {
                title: 'Yêu cầu tổng hợp lại',
                content: ev.reason ? `<div class="reason">${escapeHtml(ev.reason)}</div>` : null,
            };
        case 'clarify':
            return {
                title: 'Cần thêm thông tin từ bạn',
                content: `<div class="reason">${escapeHtml(ev.question)}</div>`,
            };
        case 'error':
            return {
                title: 'Lỗi',
                content: `<div class="reason reason-error">${escapeHtml(ev.message || 'Unknown error')}</div>`,
            };
        default:
            return null; // silently ignore unknown event types (final, done are handled elsewhere)
    }
}

// ========== Send message via SSE ==========
async function sendMessage() {
    const query = $userInput.val().trim();
    if (!query || isLoading || isTyping || !currentSessionId) return;

    hideWelcome();
    isLoading = true;
    updateSendState();

    let $group = $chatArea.find('.msg-group').last();
    if ($group.length === 0) {
        $group = $('<div class="msg-group"></div>').appendTo($chatArea);
    }
    $group.append(renderUserMessage(query));

    if ($sessionList.find(`.chat-session[data-session-id="${currentSessionId}"]`).length === 0) {
        setTimeout(loadSessions, 1500);
    }

    $userInput.val('').css('height', 'auto');
    scrollToBottom();

    const panel = createThinkingPanel();
    const $botMsg = $(`
        <div class="msg-bot">
            <div class="avatar avatar-bot">⚖️</div>
            <div class="bubble thinking-host"></div>
        </div>
    `);
    $botMsg.find('.bubble').append(panel.$el);
    $group.append($botMsg);
    scrollToBottom(true);

    let finalAnswer = null;
    let finalRefs = [];
    let gotError = null;

    try {
        const res = await fetch(`${API_BASE}/api/chat/chatbot-agentic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                use_web_fallback: isWebSearchMode,
                session_id: currentSessionId,
            }),
        });
        if (!res.ok || !res.body) {
            throw new Error(`HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // SSE frames are separated by a blank line
            let idx;
            while ((idx = buffer.indexOf('\n\n')) >= 0) {
                const frame = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                const line = frame.replace(/^data:\s*/, '').trim();
                if (!line) continue;
                let ev;
                try { ev = JSON.parse(line); }
                catch (e) { console.warn('bad SSE frame', line); continue; }

                if (ev.type === 'final') {
                    finalAnswer = ev.answer || '';
                    finalRefs = ev.references || [];
                    continue;
                }
                if (ev.type === 'done') continue;
                if (ev.type === 'error') {
                    gotError = ev.message || 'Lỗi không xác định';
                }
                panel.addStep(ev);
            }
        }
    } catch (err) {
        gotError = err.message || String(err);
    }

    if (finalAnswer != null) {
        panel.finish('Đã hoàn tất');
        typeBotAnswer($botMsg.find('.bubble'), finalAnswer, finalRefs);
    } else {
        panel.fail(gotError ? 'Lỗi xử lý' : 'Không nhận được câu trả lời');
        const msg = gotError || 'Không nhận được câu trả lời từ máy chủ.';
        $botMsg.find('.bubble').append(`<div class="error-note">${escapeHtml(msg)}</div>`);
        finishLoading();
    }
}

function typeBotAnswer($container, answer, refs) {
    isTyping = true;
    updateSendState();
    const formatted = formatBotText(answer || '');
    const $ans = $('<div class="answer-text"></div>').appendTo($container);
    const raw = answer || '';
    const CHUNK = Math.max(4, Math.ceil(raw.length / 120));
    let i = 0;
    const step = () => {
        if (i >= raw.length) {
            $ans.html(formatted);
            if (refs && refs.length > 0) appendReferences($container, refs);
            scrollToBottom();
            finishLoading();
            setTimeout(loadSessions, 300);
            return;
        }
        i += CHUNK;
        $ans.html(formatBotText(raw.slice(0, i)));
        scrollToBottom();
        setTimeout(step, 16);
    };
    step();
}

function finishLoading() {
    isLoading = false;
    isTyping = false;
    updateSendState();
    $userInput.focus();
}

// ========== Composer ==========
function updateSendState() {
    const hasText = $userInput.val().trim().length > 0;
    $sendBtn.prop('disabled', !(hasText && !isLoading && !isTyping && currentSessionId));
}

$userInput.on('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 180) + 'px';
    updateSendState();
});

$userInput.on('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

$sendBtn.on('click', sendMessage);

$webToggle.on('click', () => {
    isWebSearchMode = !isWebSearchMode;
    $webToggle.toggleClass('active', isWebSearchMode);
    if (isWebSearchMode) {
        $modeHint.html('Chế độ: <strong>RAG nội bộ + Web Search</strong> (planner tự quyết khi nào tra web)');
        $userInput.attr('placeholder', 'Nhập câu hỏi — hệ thống sẽ tra web khi cần...');
    } else {
        $modeHint.html('Chế độ: <strong>Chỉ RAG nội bộ</strong> (câu ngoài CSDL sẽ báo không biết)');
        $userInput.attr('placeholder', 'Nhập câu hỏi pháp luật của bạn...');
    }
});

// ========== Sidebar ==========
$('#toggle-sidebar').on('click', () => {
    if (window.innerWidth <= 768) $sidebar.toggleClass('open');
    else $sidebar.toggleClass('collapsed');
});

$('#new-chat').on('click', () => startNewSession());

$('#clear-current').on('click', () => {
    if (!currentSessionId) return;
    if (!confirm('Xóa cuộc trò chuyện hiện tại? Bạn sẽ được chuyển sang cuộc trò chuyện mới.')) return;
    deleteSession(currentSessionId);
});

$(document).on('click', '.suggestion-card', function() {
    const prompt = $(this).data('prompt');
    if (!prompt) return;
    $userInput.val(prompt).trigger('input').focus();
});

// ========== Boot ==========
$(document).ready(() => {
    startNewSession(true);
    loadSessions();
});
