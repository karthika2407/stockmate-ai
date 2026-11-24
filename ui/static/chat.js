/* StockMate AI - simulated chat channel.
 *
 * Every reply rendered here comes from POST /api/ask. Nothing in this file
 * composes an answer of its own: a failed request is shown as a failed
 * request.
 */
(function () {
  'use strict';

  var stream = document.getElementById('stream');
  var chips = document.getElementById('chips');
  var form = document.getElementById('composer');
  var entry = document.getElementById('entry');
  var send = document.getElementById('send');

  var QUICK = [
    'what should I buy this week',
    'show cash traps',
    'what did I find on bills',
    'help'
  ];

  var busy = false;

  function clock() {
    var d = new Date();
    var h = d.getHours();
    var m = d.getMinutes();
    return (h < 10 ? '0' : '') + h + ':' + (m < 10 ? '0' : '') + m;
  }

  function scroll() {
    stream.scrollTop = stream.scrollHeight;
  }

  /* Text goes in via textContent, so newlines survive (white-space: pre-wrap)
     and nothing in a reply can be interpreted as markup. */
  function bubble(role, text, stamp) {
    var wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    var body = document.createElement('div');
    body.className = 'bubble';
    body.textContent = text;
    wrap.appendChild(body);

    if (stamp !== false) {
      var time = document.createElement('div');
      time.className = 'time';
      time.textContent = (role === 'me' ? 'You' : 'StockMate') + ' · ' + clock();
      wrap.appendChild(time);
    }

    stream.appendChild(wrap);
    scroll();
    return wrap;
  }

  function typing() {
    var wrap = document.createElement('div');
    wrap.className = 'msg them';
    var body = document.createElement('div');
    body.className = 'bubble typing';
    body.setAttribute('aria-label', 'StockMate is replying');
    body.innerHTML = '<span></span><span></span><span></span>';
    wrap.appendChild(body);
    stream.appendChild(wrap);
    scroll();
    return wrap;
  }

  function lock(on) {
    busy = on;
    send.disabled = on;
    entry.disabled = on;
    var buttons = chips.querySelectorAll('button');
    for (var i = 0; i < buttons.length; i++) buttons[i].disabled = on;
  }

  function ask(text) {
    if (busy || !text) return;
    bubble('me', text);
    lock(true);
    var indicator = typing();

    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    }).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) {
          var why = data && data.detail ? data.detail : 'HTTP ' + r.status;
          throw new Error(typeof why === 'string' ? why : 'HTTP ' + r.status);
        }
        return data;
      }, function () {
        throw new Error('HTTP ' + r.status + ' with an unreadable body');
      });
    }).then(function (data) {
      indicator.remove();
      if (data && typeof data.reply === 'string' && data.reply.length) {
        bubble('them', data.reply);
      } else {
        bubble('fail', 'The service answered without a reply field. Nothing to show.');
      }
    }).catch(function (err) {
      indicator.remove();
      bubble('fail', 'That message could not be answered: ' + err.message +
        '\nThe reply is not shown because the service did not return one.');
    }).then(function () {
      lock(false);
      entry.focus();
    });
  }

  QUICK.forEach(function (q) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = q;
    b.addEventListener('click', function () { ask(q); });
    chips.appendChild(b);
  });

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    var text = entry.value.trim();
    if (!text) return;
    entry.value = '';
    ask(text);
  });

  bubble('sys',
    'Simulated WhatsApp channel. No WhatsApp Business account is connected. ' +
    'Send a message below, or use one of the suggested questions.', false);

  /* ----------------------------------------------------------- health */

  var dot = document.getElementById('health-dot');
  var text = document.getElementById('health-text');
  var note = document.getElementById('channel-note');
  var side = document.getElementById('side-health');

  function kv(label, value) {
    var row = document.createElement('div');
    row.className = 'kv';
    var l = document.createElement('span');
    l.textContent = label;
    var v = document.createElement('b');
    v.textContent = value;
    row.appendChild(l);
    row.appendChild(v);
    return row;
  }

  fetch('/api/health').then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function (h) {
    dot.className = 'dot live';
    text.textContent = 'Service ' + (h.status || 'ok') + ' · v' + (h.version || '?');
    if (h.channel) note.textContent = 'Channel: ' + h.channel;
    side.textContent = '';
    side.appendChild(kv('Status', String(h.status)));
    side.appendChild(kv('Version', String(h.version)));
    side.appendChild(kv('SKUs tracked', String(h.skus)));
    side.appendChild(kv('Bills processed', String(h.bills_seen)));
    side.appendChild(kv('Stored computations', String(h.ledger_entries)));
  }).catch(function (err) {
    dot.className = 'dot down';
    text.textContent = 'Service unreachable';
    note.textContent = 'Channel status unavailable while the service is unreachable.';
    side.innerHTML = '';
    var box = document.createElement('div');
    box.className = 'error';
    box.textContent = 'GET /api/health failed (' + err.message +
      '). Messages will not be answered until the service is running.';
    side.appendChild(box);
  });

  entry.focus();
})();
