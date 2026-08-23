#!/usr/bin/env python3
"""
ns_treatments.py — Herramienta para LISTAR y BORRAR tratamientos en Nightscout.

Sin dependencias externas (solo Python 3). Pensada para correr en TU máquina,
que tiene internet hacia Nightscout.

⚠️ SEGURIDAD: borrar es permanente. El script SIEMPRE muestra qué va a borrar y
pide confirmación (salvo que pases --yes). Son datos de salud: revisá bien.

CONFIGURACIÓN (por variables de entorno o argumentos):
  NIGHTSCOUT_URL         (o --url)     ej: https://p01--nightscout--nyw72jfcywg7.code.run
  NIGHTSCOUT_API_SECRET  (o --secret)  tu API secret de Nightscout (texto plano;
                                       el script calcula el SHA1 solo)

USO:
  # Listar los últimos 20 tratamientos (con su _id):
  python ns_treatments.py list

  # Listar las últimas 48 horas:
  python ns_treatments.py list --hours 48

  # Borrar uno o varios por _id (pide confirmación):
  python ns_treatments.py delete 66c9... 66ca...

  # Borrar sin preguntar (¡cuidado!):
  python ns_treatments.py delete 66c9... --yes

  # Ver qué borraría el filtro de "Carb Correction sin gramos" (las fotos vacías):
  python ns_treatments.py list --empty-carbs

Ejemplos con credenciales por argumento (si no usás variables de entorno):
  python ns_treatments.py --url https://TU-NS --secret TU_SECRET list
"""

import os
import sys
import json
import hashlib
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

TZ_AR = timezone(timedelta(hours=-3))
DEFAULT_URL = "https://p01--nightscout--nyw72jfcywg7.code.run"


def get_config(args):
    url = (args.url or os.environ.get("NIGHTSCOUT_URL") or DEFAULT_URL).rstrip("/")
    secret = args.secret or os.environ.get("NIGHTSCOUT_API_SECRET") or ""
    if not secret:
        sys.exit("ERROR: falta el API secret. Pasá --secret o seteá NIGHTSCOUT_API_SECRET.")
    return url, hashlib.sha1(secret.encode("utf-8")).hexdigest()


def api_request(url, secret_sha1, method="GET"):
    req = urllib.request.Request(url, method=method)
    req.add_header("api-secret", secret_sha1)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")
    except Exception as e:
        return None, str(e)


def fmt_time(created):
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(TZ_AR)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return (created or "?")[:16]


def describe(t):
    parts = [t.get("eventType", "?")]
    if t.get("insulin"):
        parts.append(f"💉{t['insulin']}U")
    if t.get("carbs"):
        parts.append(f"🍞{t['carbs']}g")
    if t.get("duration"):
        parts.append(f"⏱{t['duration']}min")
    if t.get("glucose"):
        parts.append(f"🩸{t['glucose']}")
    notes = t.get("notes")
    if notes:
        parts.append(f"— {notes[:40]}")
    who = t.get("enteredBy", "")
    return " ".join(parts) + (f"  ({who})" if who else "")


def cmd_list(url, secret, args):
    count = args.count
    endpoint = f"{url}/api/v1/treatments.json?count={count}"
    if args.hours:
        since = (datetime.now(timezone.utc) - timedelta(hours=args.hours)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z")
        endpoint = (f"{url}/api/v1/treatments.json"
                    f"?find[created_at][$gte]={since}&count=200")
    status, data = api_request(endpoint, secret)
    if status != 200 or not isinstance(data, list):
        sys.exit(f"ERROR al listar (HTTP {status}): {data}")

    if args.empty_carbs:
        data = [t for t in data
                if t.get("eventType") == "Carb Correction" and not t.get("carbs")]
        if not data:
            print("No hay 'Carb Correction' sin gramos. Nada para limpiar.")
            return

    if not data:
        print("No hay tratamientos en ese rango.")
        return

    print(f"\n{len(data)} tratamiento(s):\n")
    for t in data:
        print(f"  {fmt_time(t.get('created_at'))}  {describe(t)}")
        print(f"      _id: {t.get('_id')}")
    print("\nPara borrar: python ns_treatments.py delete <_id> [<_id> ...]")


def cmd_delete(url, secret, args):
    ids = args.ids
    if not ids:
        sys.exit("Pasá al menos un _id. Ej: python ns_treatments.py delete 66c9...")

    # Mostrar qué se va a borrar (traemos el detalle de cada uno)
    print("\nSe van a BORRAR estos tratamientos:\n")
    to_delete = []
    for _id in ids:
        status, data = api_request(f"{url}/api/v1/treatments.json?find[_id]={_id}", secret)
        if status == 200 and isinstance(data, list) and data:
            t = data[0]
            print(f"  {fmt_time(t.get('created_at'))}  {describe(t)}")
            print(f"      _id: {_id}")
            to_delete.append(_id)
        else:
            print(f"  ⚠️ No encontré el _id {_id} (lo salteo).")
    if not to_delete:
        sys.exit("\nNada válido para borrar.")

    if not args.yes:
        resp = input(f"\n¿Confirmás borrar {len(to_delete)} registro(s)? (escribí 'si'): ").strip().lower()
        if resp not in ("si", "sí", "s", "yes", "y"):
            sys.exit("Cancelado. No se borró nada.")

    ok = 0
    for _id in to_delete:
        status, data = api_request(f"{url}/api/v1/treatments/{_id}", secret, method="DELETE")
        if status in (200, 204):
            print(f"  ✅ borrado {_id}")
            ok += 1
        else:
            print(f"  ❌ error borrando {_id} (HTTP {status}): {data}")
    print(f"\nListo: {ok}/{len(to_delete)} borrados.")


def main():
    p = argparse.ArgumentParser(description="Listar y borrar tratamientos de Nightscout.")
    p.add_argument("--url", help="URL de Nightscout (o env NIGHTSCOUT_URL)")
    p.add_argument("--secret", help="API secret en texto plano (o env NIGHTSCOUT_API_SECRET)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="Listar tratamientos con su _id")
    pl.add_argument("--count", type=int, default=20, help="Cuántos traer (default 20)")
    pl.add_argument("--hours", type=int, help="Traer los de las últimas N horas")
    pl.add_argument("--empty-carbs", action="store_true",
                    help="Mostrar solo 'Carb Correction' sin gramos (fotos vacías)")

    pd = sub.add_parser("delete", help="Borrar tratamiento(s) por _id")
    pd.add_argument("ids", nargs="+", help="Uno o más _id a borrar")
    pd.add_argument("--yes", action="store_true", help="No preguntar confirmación")

    args = p.parse_args()
    url, secret = get_config(args)
    if args.cmd == "list":
        cmd_list(url, secret, args)
    elif args.cmd == "delete":
        cmd_delete(url, secret, args)


if __name__ == "__main__":
    main()
