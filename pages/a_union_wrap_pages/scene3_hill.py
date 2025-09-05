import os
import tempfile
import requests
import streamlit as st

import json
import math
import io
from PIL import Image, ImageDraw
import requests

#!/usr/bin/env python3
"""


提供函数 fetch_and_draw_satellite(geojson_path, out_dir, z=16, style=6, padding_ratio=0.06, max_size=(2048,2048))：
 - 自动调整 zoom，使得最终拼接图像不会超过 max_size（默认 2048x2048）
 - 读取 GeoJSON，提取矩形/多边形坐标
 - 计算覆盖所有矩形的扩展 bbox
 - 下载并拼接高德卫星瓦片
 - 绘制矩形到图像
 - 返回输出图像路径

附带测试用例。
"""

TILE_URL = "https://webst0{srv}.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}&scl=1&ltype=11"


def lonlat_to_global_pixel(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * 256.0
    sin_lat = math.sin(math.radians(lat))
    sin_lat = min(max(sin_lat, -0.9999), 0.9999)
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n * 256.0
    return x, y


def fetch_tile(x, y, z, style=6, timeout=10):
    srv = (x + y) % 4 + 1
    url = TILE_URL.format(srv=srv, style=style, x=x, y=y, z=z)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def stitch_tiles(xmin, xmax, ymin, ymax, z, style=6, placeholder_color=(200,200,200)):
    w_tiles = xmax - xmin + 1
    h_tiles = ymax - ymin + 1
    out_w = w_tiles * 256
    out_h = h_tiles * 256
    canvas = Image.new("RGB", (out_w, out_h), color=placeholder_color)

    for ix, tx in enumerate(range(xmin, xmax+1)):
        for iy, ty in enumerate(range(ymin, ymax+1)):
            try:
                tile = fetch_tile(tx, ty, z, style=style)
                canvas.paste(tile, (ix*256, iy*256))
            except Exception as e:
                print(f"warning: failed to fetch tile {tx},{ty}, z={z}: {e}")
    left_px = xmin * 256
    top_px = ymin * 256
    return canvas, left_px, top_px


def collect_coords_from_geojson(geojson_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    features = data.get('features', []) if isinstance(data, dict) else []
    all_polygons = []

    for feat in features:
        geom = feat.get('geometry') if isinstance(feat, dict) else None
        if not geom:
            continue
        gtype = geom.get('type')
        coords = geom.get('coordinates')
        if gtype == 'Polygon':
            ring = coords[0]
            all_polygons.append(ring)
        elif gtype == 'MultiPolygon':
            for poly in coords:
                ring = poly[0]
                all_polygons.append(ring)
        elif gtype == 'Point':
            props = feat.get('properties', {})
            if 'bbox' in props:
                minx, miny, maxx, maxy = props['bbox']
                ring = [(minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy),(minx,miny)]
                all_polygons.append(ring)
            else:
                lon, lat = coords
                all_polygons.append([(lon,lat)])
    return all_polygons


def bbox_of_polygons(polygons):
    min_lon = 180.0
    min_lat = 90.0
    max_lon = -180.0
    max_lat = -90.0
    for poly in polygons:
        for p in poly:
            lon, lat = p
            min_lon = min(min_lon, lon)
            min_lat = min(min_lat, lat)
            max_lon = max(max_lon, lon)
            max_lat = max(max_lat, lat)
    return min_lon, min_lat, max_lon, max_lat


def expand_bbox(min_lon, min_lat, max_lon, max_lat, padding_ratio=0.05):
    lon_pad = (max_lon - min_lon) * padding_ratio or 0.0005
    lat_pad = (max_lat - min_lat) * padding_ratio or 0.0005
    return min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad


def draw_polygons_on_image(base_img, polygons, z, left_px, top_px, outline=(0,0,0), width=3, fill=(255,255,0,255)):
    """
    polygons: 多边形列表
    outline: 边框颜色, 黑色
    fill: 填充颜色, 黄色不透明
    """
    draw = ImageDraw.Draw(base_img, 'RGBA')
    for poly in polygons:
        pxs = []
        for lon, lat in poly:
            gx, gy = lonlat_to_global_pixel(lon, lat, z)
            rx = gx - left_px
            ry = gy - top_px
            pxs.append((rx, ry))
        if len(pxs) == 1:
            x, y = pxs[0]
            r = max(4, width*2)
            draw.line((x-r, y, x+r, y), fill=outline+(255,), width=width)
            draw.line((x, y-r, x, y+r), fill=outline+(255,), width=width)
        else:
            draw.polygon(pxs, outline=outline+(255,), fill=fill)
    return base_img




def adjust_zoom_to_fit(min_lon, min_lat, max_lon, max_lat, z, max_size):
    """自动调整 zoom 直到拼接图像不超过 max_size"""
    while z > 3:
        left_px, top_px = lonlat_to_global_pixel(min_lon, max_lat, z)
        right_px, bottom_px = lonlat_to_global_pixel(max_lon, min_lat, z)
        w = right_px - left_px
        h = bottom_px - top_px
        if w <= max_size[0] and h <= max_size[1]:
            break
        z -= 1
    return z


def fetch_and_draw_satellite(geojson_path, out_dir, z=16, style=6, padding_ratio=0.06, max_size=(2048,2048)):
    polys = collect_coords_from_geojson(geojson_path)
    if not polys:
        raise ValueError(f'No polygons found in {geojson_path}')

    min_lon, min_lat, max_lon, max_lat = bbox_of_polygons(polys)
    min_lon, min_lat, max_lon, max_lat = expand_bbox(min_lon, min_lat, max_lon, max_lat, padding_ratio)

    # 自动调整 zoom
    z = adjust_zoom_to_fit(min_lon, min_lat, max_lon, max_lat, z, max_size)
    print(f"Using zoom {z}")

    left_px_f, top_px_f = lonlat_to_global_pixel(min_lon, max_lat, z)
    right_px_f, bottom_px_f = lonlat_to_global_pixel(max_lon, min_lat, z)

    xmin_tile = int(math.floor(left_px_f / 256.0))
    ymin_tile = int(math.floor(top_px_f / 256.0))
    xmax_tile = int(math.floor(right_px_f / 256.0))
    ymax_tile = int(math.floor(bottom_px_f / 256.0))

    base_img, left_px, top_px = stitch_tiles(xmin_tile, xmax_tile, ymin_tile, ymax_tile, z, style=style)

    sat_out = os.path.join(out_dir, 'satellite_raw.png')
    base_img.save(sat_out)

    overlaid = base_img.copy()
    overlaid = draw_polygons_on_image(overlaid, polys, z, left_px, top_px)
    over_out = os.path.join(out_dir, 'satellite_with_rects.png')
    overlaid.save(over_out)

    return sat_out, over_out

def run_scene3():
    st.title("遥感采样地图位置信息封装")

    url = ("https://gaoyuan-49d0.obs.cn-north-4.myhuaweicloud.com/"
           "%E7%9F%B3%E5%86%B0%E5%B7%9D%E6%95%B0%E6%8D%AE-%E9%81%A5%E6%84%9F%2B%E6%97%A0%E4%BA%BA%E6%9C%BA/"
           "output_pos.geojson")

    with st.spinner("正在加载并绘制轨迹图..."):
        try:
            # 1. 下载 geojson 到临时文件
            tmpdir = tempfile.mkdtemp(prefix="scene3_")
            geojson_path = os.path.join(tmpdir, "input.geojson")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(geojson_path, "wb") as f:
                f.write(resp.content)

            # 2. 调用 fetch_and_draw_satellite，返回生成的图片路径
            _, over_path = fetch_and_draw_satellite(
                geojson_path=geojson_path,
                out_dir=tmpdir,
                z=16,
                style=6,
                padding_ratio=0.06,
                max_size=(2048, 2048),
            )

            # 3. 用 st.image 展示 with_rects 图
            over_img = Image.open(over_path)
            st.image(over_img, caption="轨迹图 (with_rects)", use_container_width=True)

        except Exception as e:
            st.error(f"绘制失败: {e}")
