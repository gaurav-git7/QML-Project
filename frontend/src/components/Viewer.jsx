import { useCallback, useEffect, useRef, useState } from 'react'
import { Icon } from './Icons.jsx'

/** Radiograph viewer: window level / width, invert, zoom (wheel or buttons) and pan (drag). Display only - the models
 *  always see the original image. `leftControls` goes in the toolbar, `empty` is shown when no image is loaded. */
export default function Viewer({ src, onSize, leftControls, empty, overlay }) {
  const canvasRef = useRef(null)
  const stageRef = useRef(null)
  const original = useRef(null)
  const drag = useRef(null)
  const [level, setLevel] = useState(50)
  const [width, setWidth] = useState(100)
  const [invert, setInvert] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })

  const render = useCallback(() => {
    const o = original.current
    const c = canvasRef.current
    if (!o || !c) return
    const lo = Math.max(0, level / 100 - width / 200)
    const hi = Math.max(Math.min(1, level / 100 + width / 200), lo + 0.001)
    const lut = new Uint8ClampedArray(256)
    for (let v = 0; v < 256; v++) {
      let t = (v / 255 - lo) / (hi - lo)
      t = Math.min(1, Math.max(0, t))
      lut[v] = (invert ? 1 - t : t) * 255
    }
    const out = new ImageData(o.width, o.height)
    for (let i = 0; i < o.data.length; i += 4) {
      const g = lut[o.data[i]]
      out.data[i] = out.data[i + 1] = out.data[i + 2] = g
      out.data[i + 3] = 255
    }
    c.getContext('2d').putImageData(out, 0, 0)
  }, [level, width, invert])

  const resetView = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); setLevel(50); setWidth(100); setInvert(false) }, [])

  // load a new image
  useEffect(() => {
    original.current = null
    resetView()
    if (!src) return
    const img = new Image()
    img.onload = () => {
      const s = Math.min(1, 1600 / Math.max(img.width, img.height))
      const w = Math.round(img.width * s), h = Math.round(img.height * s)
      const off = document.createElement('canvas')
      off.width = w; off.height = h
      const ctx = off.getContext('2d')
      ctx.drawImage(img, 0, 0, w, h)
      original.current = ctx.getImageData(0, 0, w, h)
      const c = canvasRef.current
      if (c) { c.width = w; c.height = h }
      onSize && onSize([img.naturalWidth, img.naturalHeight])
      render()
    }
    img.src = src
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src])

  useEffect(() => { render() }, [render])

  // keyboard shortcuts are dispatched by the page as window events
  useEffect(() => {
    const inv = () => setInvert((v) => !v)
    const rst = () => resetView()
    const zin = () => setZoom((z) => Math.min(8, z * 1.25))
    const zout = () => setZoom((z) => Math.max(0.5, z / 1.25))
    window.addEventListener('viewer-invert', inv); window.addEventListener('viewer-reset', rst)
    window.addEventListener('viewer-zoom-in', zin); window.addEventListener('viewer-zoom-out', zout)
    return () => { window.removeEventListener('viewer-invert', inv); window.removeEventListener('viewer-reset', rst); window.removeEventListener('viewer-zoom-in', zin); window.removeEventListener('viewer-zoom-out', zout) }
  }, [resetView])

  // wheel zoom (non-passive). On narrow screens the page itself scrolls, so zoom only with Ctrl there.
  useEffect(() => {
    const el = stageRef.current
    if (!el) return
    const onWheel = (e) => {
      if (!(e.ctrlKey || window.innerWidth >= 1020)) return
      e.preventDefault()
      setZoom((z) => Math.min(8, Math.max(0.5, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))))
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const down = (e) => { if (!src) return; drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }; e.currentTarget.setPointerCapture(e.pointerId) }
  const move = (e) => { if (drag.current) setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y }) }
  const up = () => { drag.current = null }

  return (
    <>
      <div className="viewer-bar">
        <div className="grp">{leftControls}</div>
        <div className="grp" role="toolbar" aria-label="Viewer tools">
          <button className="btn dark icon" title="Zoom in (+)" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(8, z * 1.25))} disabled={!src}><Icon name="plus" size={16} /></button>
          <button className="btn dark icon" title="Zoom out (-)" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(0.5, z / 1.25))} disabled={!src}><Icon name="minus" size={16} /></button>
          <button className="btn dark icon" title="Invert (I)" aria-label="Invert image" aria-pressed={invert} onClick={() => setInvert((v) => !v)} disabled={!src}><Icon name="invert" size={16} /></button>
          <button className="btn dark icon" title="Reset view (R)" aria-label="Reset view" onClick={resetView} disabled={!src}><Icon name="reset" size={16} /></button>
        </div>
      </div>
      <div className="viewer-stage" ref={stageRef} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        <canvas ref={canvasRef} className={src ? '' : 'hidden'} style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }} aria-label="Chest radiograph" role="img" />
        {!src && empty}
        {src && overlay && <div className="viewer-overlay">{overlay}{zoom !== 1 ? ` · ${Math.round(zoom * 100)}%` : ''}</div>}
      </div>
      <div className="viewer-foot">
        <div>
          <label htmlFor="wl"><span>Window level</span><span>{level}</span></label>
          <input id="wl" type="range" min="0" max="100" value={level} onChange={(e) => setLevel(+e.target.value)} disabled={!src} />
        </div>
        <div>
          <label htmlFor="ww"><span>Window width</span><span>{width}</span></label>
          <input id="ww" type="range" min="1" max="100" value={width} onChange={(e) => setWidth(+e.target.value)} disabled={!src} />
        </div>
      </div>
    </>
  )
}
