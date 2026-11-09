import { useMemo, useState } from 'react'

/** columns: [{ key, label, render?(row), sortValue?(row) }]. Sortable (keyboard accessible), optional paging and row activation. */
export default function DataTable({ columns, rows, pageSize = 0, pageSizes, onRowClick, rowLabel, initialSort, emptyText = 'No rows.' }) {
  const [sort, setSort] = useState(initialSort || null)
  const [page, setPage] = useState(0)
  const [size, setSize] = useState(pageSize)

  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    const val = (r) => (col?.sortValue ? col.sortValue(r) : r[sort.key])
    return [...rows].sort((a, b) => {
      const x = val(a), y = val(b)
      if (x == null) return 1
      if (y == null) return -1
      return (x > y ? 1 : x < y ? -1 : 0) * (sort.dir === 'asc' ? 1 : -1)
    })
  }, [rows, sort, columns])

  const pages = size ? Math.max(1, Math.ceil(sorted.length / size)) : 1
  const cur = Math.min(page, pages - 1)
  const shown = size ? sorted.slice(cur * size, cur * size + size) : sorted
  const toggle = (key) => { setPage(0); setSort((s) => (s && s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' })) }

  return (
    <>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key} aria-sort={sort?.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}>
                  <button onClick={() => toggle(c.key)} title={`Sort by ${c.label}`}>{c.label}{sort?.key === c.key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''}</button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={r.id ?? i} className={onRowClick ? 'click' : ''} tabIndex={onRowClick ? 0 : undefined}
                aria-label={onRowClick && rowLabel ? rowLabel(r) : undefined}
                onClick={onRowClick ? () => onRowClick(r) : undefined}
                onKeyDown={onRowClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onRowClick(r) } } : undefined}>
                {columns.map((c) => <td key={c.key}>{c.render ? c.render(r) : r[c.key]}</td>)}
              </tr>
            ))}
            {!shown.length && <tr><td colSpan={columns.length} style={{ textAlign: 'center', color: 'var(--muted)', padding: 26 }}>{emptyText}</td></tr>}
          </tbody>
        </table>
      </div>
      {size > 0 && (
        <div className="pager">
          <span>{sorted.length.toLocaleString()} rows · page {cur + 1} of {pages}</span>
          <span className="actions">
            {pageSizes && (
              <label className="small">Rows per page{' '}
                <select className="select" style={{ width: 'auto', padding: '4px 8px' }} value={size} onChange={(e) => { setSize(+e.target.value); setPage(0) }}>
                  {pageSizes.map((p) => <option key={p}>{p}</option>)}
                </select>
              </label>
            )}
            <button className="btn sm" disabled={cur === 0} onClick={() => setPage(cur - 1)}>Previous</button>
            <button className="btn sm" disabled={cur >= pages - 1} onClick={() => setPage(cur + 1)}>Next</button>
          </span>
        </div>
      )}
    </>
  )
}
