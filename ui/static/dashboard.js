/* StockMate AI - dashboard.
 *
 * Everything on this page is rendered from /api/health, /api/dashboard and
 * /api/plan. There is no sample data in this file and no fallback content: if a
 * request fails, the panel says so and shows nothing else.
 */
(function () {
  'use strict';

  /* ------------------------------------------------------------ helpers */

  /* Indian digit grouping: last three digits, then pairs. Used only when the
     API hands us paise without a formatted string of its own. */
  function rupees(paise) {
    if (paise === null || paise === undefined || isNaN(paise)) return null;
    var neg = paise < 0;
    var whole = Math.floor(Math.abs(paise) / 100);
    var fraction = String(Math.abs(paise) % 100);
    if (fraction.length < 2) fraction = '0' + fraction;

    var digits = String(whole);
    var grouped;
    if (digits.length <= 3) {
      grouped = digits;
    } else {
      var tail = digits.slice(-3);
      var head = digits.slice(0, -3);
      var parts = [];
      while (head.length > 2) {
        parts.unshift(head.slice(-2));
        head = head.slice(0, -2);
      }
      if (head.length) parts.unshift(head);
      grouped = parts.join(',') + ',' + tail;
    }
    return (neg ? '-' : '') + 'Rs ' + grouped + '.' + fraction;
  }

  /* Prefer the string the API formatted; fall back to formatting the paise. */
  function money(formatted, paise) {
    if (typeof formatted === 'string' && formatted.length) return formatted;
    return rupees(paise);
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function errorBox(where, err) {
    var box = el('div', 'error');
    box.appendChild(el('strong', null, where + ' could not be loaded.'));
    box.appendChild(document.createTextNode(
      err.message + '. Nothing is shown for this panel rather than showing a figure ' +
      'the service did not return. Check that the API is running, then reload.'));
    return box;
  }

  function emptyBox(text) {
    return el('div', 'empty', text);
  }

  function slug(value) {
    return String(value || '').toLowerCase().replace(/[^a-z_]/g, '');
  }

  function asJson(response) {
    return response.json().then(function (data) {
      if (!response.ok) {
        var detail = data && data.detail;
        throw new Error(typeof detail === 'string' ? detail : 'HTTP ' + response.status);
      }
      return data;
    }, function () {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      throw new Error('the response was not valid JSON');
    });
  }

  /* ------------------------------------------------------------- health */

  var dot = document.getElementById('health-dot');
  var healthText = document.getElementById('health-text');
  var channelNote = document.getElementById('channel-note');

  fetch('/api/health').then(asJson).then(function (h) {
    dot.className = 'dot live';
    healthText.textContent = 'Service ' + (h.status || 'ok') + ' · v' + (h.version || '?');
    if (h.channel) channelNote.textContent = 'Channel: ' + h.channel;
  }).catch(function () {
    dot.className = 'dot down';
    healthText.textContent = 'Service unreachable';
    channelNote.textContent = 'Channel status unavailable while the service is unreachable.';
  });

  /* ---------------------------------------------------------- findings */

  /* A finding row opens to show the one-line working from the audit trail.
     The dashboard payload may or may not carry an audit trail; when it does
     not, the row states which computation the amount is recorded against
     instead of inventing a working line. */
  function findingRow(finding, trail) {
    var item = el('div', 'finding');

    var button = el('button', 'finding-row');
    button.type = 'button';
    button.setAttribute('aria-expanded', 'false');

    button.appendChild(el('span', 'caret', '+'));

    var middle = el('span');
    middle.appendChild(el('span', 'ftitle', finding.title || finding.kind || 'Finding'));
    if (finding.text) middle.appendChild(el('span', 'ftext', finding.text));
    button.appendChild(middle);

    var amount = money(finding.amount, finding.amount_paise);
    var amountNode;
    if (amount) {
      amountNode = el('span', 'money famount ' + slug(finding.bucket), amount);
    } else {
      amountNode = el('span', 'famount none', 'no amount');
    }
    button.appendChild(amountNode);

    button.appendChild(el('span', 'pill ' + slug(finding.severity),
      finding.severity ? String(finding.severity) : 'unrated'));

    item.appendChild(button);

    var working = el('div', 'working');
    working.appendChild(el('h4', null, 'Working'));

    var entry = trail && finding.finding_id ? trail[finding.finding_id] : null;
    if (entry && entry.working) {
      working.appendChild(el('div', 'line', entry.working));
    } else if (finding.computation_id) {
      working.appendChild(el('div', 'unavailable',
        'The amount is recorded against stored computation ' + finding.computation_id +
        '. This payload does not include the audit trail, so the one-line working is not ' +
        'repeated here; it is returned with the bill that produced this finding.'));
    } else if (amount) {
      working.appendChild(el('div', 'unavailable',
        'No computation reference came with this finding.'));
    } else {
      working.appendChild(el('div', 'unavailable',
        'This finding carries no amount, so there is no computation behind it.'));
    }

    var meta = el('div', 'meta');
    if (finding.kind) meta.appendChild(el('span', null, 'kind: ' + finding.kind));
    if (finding.bucket) meta.appendChild(el('span', null, 'bucket: ' + finding.bucket));
    if (finding.amount_paise !== null && finding.amount_paise !== undefined) {
      meta.appendChild(el('span', null, 'paise: ' + finding.amount_paise));
    }
    if (finding.computation_id) {
      var ref = el('span', null, 'computation: ');
      ref.appendChild(el('code', null, finding.computation_id));
      meta.appendChild(ref);
    }
    if (finding.explained_by) meta.appendChild(el('span', null, 'explained by: ' + finding.explained_by));
    if (meta.childNodes.length) working.appendChild(meta);

    item.appendChild(working);

    button.addEventListener('click', function () {
      var open = item.classList.toggle('open');
      button.setAttribute('aria-expanded', open ? 'true' : 'false');
      button.querySelector('.caret').textContent = open ? '−' : '+';
    });

    return item;
  }

  function findingList(target, findings, trail, emptyText) {
    target.innerHTML = '';
    if (!findings || !findings.length) {
      var blank = el('div', 'card');
      blank.appendChild(emptyBox(emptyText));
      target.appendChild(blank);
      return;
    }
    var card = el('div', 'card finding-list');
    findings.forEach(function (f) { card.appendChild(findingRow(f, trail)); });
    target.appendChild(card);
  }

  function statCard(kind, label, value, note) {
    var card = el('div', 'card stat ' + kind);
    card.appendChild(el('div', 'label', label));
    card.appendChild(el('div', 'value money', value === null ? 'not reported' : value));
    card.appendChild(el('div', 'note', note));
    return card;
  }

  /* --------------------------------------------------------- dashboard */

  var statsBox = document.getElementById('stats');
  var billBox = document.getElementById('bill-findings');
  var trapBox = document.getElementById('cash-traps');
  var trapTotalsBox = document.getElementById('trap-totals');
  var asOf = document.getElementById('as-of');

  fetch('/api/dashboard').then(asJson).then(function (d) {
    var head = d.headline || {};

    var trail = null;
    if (Array.isArray(d.audit_trail)) {
      trail = {};
      d.audit_trail.forEach(function (row) {
        if (row && row.finding && row.finding.finding_id) {
          trail[row.finding.finding_id] = row;
        }
      });
    }

    var stats = el('div', 'stats');
    stats.appendChild(statCard('found', 'Money found',
      money(head.found, head.found_paise),
      'Overcharges and billing errors caught on supplier bills.'));
    stats.appendChild(statCard('released', 'Money released',
      money(head.released, head.released_paise),
      'Cash recoverable from slow-moving and dead stock.'));
    stats.appendChild(statCard('protected', 'Money protected',
      money(head.protected, head.protected_paise),
      'Value at risk of being lost at expiry, flagged in time.'));
    statsBox.innerHTML = '';
    statsBox.appendChild(stats);

    var bits = [];
    if (d.as_of) bits.push('as of ' + d.as_of);
    if (d.bills_seen !== undefined) bits.push(d.bills_seen + ' bills seen');
    if (d.skus !== undefined) bits.push(d.skus + ' SKUs');
    asOf.textContent = bits.join(' · ');

    findingList(billBox, d.bill_findings, trail,
      'No bill findings yet. Send a bill to POST /api/bills and the audit will appear here.');

    var totals = d.trap_totals || {};
    trapTotalsBox.innerHTML = '';
    if (totals.total_paise !== undefined && totals.total_paise !== null) {
      var bar = el('div', 'card budget-bar');
      [['Slow moving', totals.slow_value_paise],
       ['Dead stock', totals.dead_value_paise],
       ['Expiring', totals.expiry_value_paise],
       ['Total trapped', totals.total_paise]].forEach(function (pair) {
        if (pair[1] === undefined || pair[1] === null) return;
        var cell = el('div', null, pair[0]);
        cell.appendChild(el('b', 'money', rupees(pair[1])));
        bar.appendChild(cell);
      });
      bar.style.marginBottom = '12px';
      trapTotalsBox.appendChild(bar);
    }

    findingList(trapBox, d.cash_traps, trail,
      'No cash traps found. Nothing in stock is currently slow, dead or close to expiry ' +
      'by the configured thresholds.');

  }).catch(function (err) {
    statsBox.innerHTML = '';
    statsBox.appendChild(errorBox('The headline figures', err));
    billBox.innerHTML = '';
    billBox.appendChild(errorBox('Bill findings', err));
    trapTotalsBox.innerHTML = '';
    trapBox.innerHTML = '';
    trapBox.appendChild(errorBox('Cash traps', err));
    asOf.textContent = '';
  });

  /* ------------------------------------------------------------- plan */

  var planBox = document.getElementById('plan');
  var budget = document.getElementById('budget');
  var service = document.getElementById('service');
  var review = document.getElementById('review');

  var DECISIONS = { buy: 'Buy', buy_less: 'Buy less', dont_buy: 'Do not buy' };
  var pending = null;
  var token = 0;

  function planTable(p) {
    var frag = document.createDocumentFragment();

    var bar = el('div', 'budget-bar');
    [['Budget', p.budget_paise], ['Planned spend', p.spend_paise], ['Headroom', p.headroom_paise]]
      .forEach(function (pair) {
        var cell = el('div', null, pair[0]);
        cell.appendChild(el('b', 'money', rupees(pair[1])));
        bar.appendChild(cell);
      });
    var cut = el('div', 'cut', 'Lines reduced');
    cut.appendChild(el('b', null, String(p.lines_cut) + ' of ' + String(p.lines)));
    bar.appendChild(cut);
    if (p.method) {
      var method = el('div', null, 'Allocation');
      method.appendChild(el('b', null, String(p.method).replace(/_/g, ' ')));
      bar.appendChild(method);
    }
    frag.appendChild(bar);

    if (p.explanation) frag.appendChild(el('div', 'plan-note', p.explanation));

    var rows = p.suggestions || [];
    if (!rows.length) {
      frag.appendChild(emptyBox('The planner returned no lines for this budget and service level.'));
      return frag;
    }

    var scroll = el('div', 'table-scroll');
    var table = el('table', 'plan');
    var thead = el('thead');
    var hr = el('tr');
    ['Item', 'Decision', 'Qty', 'Cash required', 'Reorder point', 'Cover days', 'Daily forecast', 'Reason']
      .forEach(function (h) { hr.appendChild(el('th', null, h)); });
    thead.appendChild(hr);
    table.appendChild(thead);

    var tbody = el('tbody');
    rows.forEach(function (s) {
      var tr = el('tr');

      var item = el('td');
      item.appendChild(el('div', null, s.name || s.sku_id));
      if (s.sku_id) item.appendChild(el('div', 'sku', s.sku_id));
      tr.appendChild(item);

      var decision = el('td');
      decision.appendChild(el('span', 'pill ' + slug(s.decision),
        DECISIONS[s.decision] || String(s.decision || 'unknown')));
      tr.appendChild(decision);

      tr.appendChild(el('td', 'num', s.suggest_qty !== undefined ? String(s.suggest_qty) : '-'));
      tr.appendChild(el('td', 'num money', rupees(s.cash_required_paise) || '-'));
      tr.appendChild(el('td', 'num', s.reorder_point !== undefined ? String(s.reorder_point) : '-'));
      tr.appendChild(el('td', 'num',
        s.cover_days === null || s.cover_days === undefined ? 'none' : Number(s.cover_days).toFixed(1)));
      tr.appendChild(el('td', 'num',
        s.forecast_daily === null || s.forecast_daily === undefined ? '-' : Number(s.forecast_daily).toFixed(2)));
      tr.appendChild(el('td', 'reason', s.reason || ''));

      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    scroll.appendChild(table);
    frag.appendChild(scroll);
    return frag;
  }

  function loadPlan() {
    var amount = Number(budget.value);
    if (!isFinite(amount) || amount < 0) {
      planBox.innerHTML = '';
      planBox.appendChild(emptyBox('Enter a cash budget of zero or more to see a plan.'));
      return;
    }
    var days = Number(review.value);
    if (!isFinite(days) || days < 1) days = 7;

    var mine = ++token;
    planBox.innerHTML = '';
    planBox.appendChild(el('div', 'loading', 'Planning against ' + rupees(Math.round(amount * 100)) + '…'));

    var query = '/api/plan?budget=' + encodeURIComponent(amount) +
                '&service_level=' + encodeURIComponent(service.value) +
                '&review_days=' + encodeURIComponent(days);

    fetch(query).then(asJson).then(function (p) {
      if (mine !== token) return;
      planBox.innerHTML = '';
      planBox.appendChild(planTable(p));
    }).catch(function (err) {
      if (mine !== token) return;
      planBox.innerHTML = '';
      planBox.appendChild(errorBox('The order plan', err));
    });
  }

  function schedule() {
    if (pending) clearTimeout(pending);
    pending = setTimeout(loadPlan, 300);
  }

  budget.addEventListener('input', schedule);
  review.addEventListener('input', schedule);
  service.addEventListener('change', loadPlan);
  loadPlan();
})();
