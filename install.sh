#!/bin/sh
set -eu

version=${NAKY_VERSION:-v0.1.0}
prefix=${NAKY_INSTALL_DIR:-"${HOME}/.local"}
repository=${NAKY_RELEASE_REPOSITORY:-superlinked/naky}

if ! printf '%s\n' "$version" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "invalid NAKY_VERSION: $version" >&2
  exit 2
fi
case "$prefix" in
  /*) ;;
  *) echo "NAKY_INSTALL_DIR must be an absolute path" >&2; exit 2 ;;
esac

for command in curl sha256sum tar install mktemp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is missing: $command" >&2
    exit 1
  fi
done

case "$(uname -s):$(uname -m)" in
  Linux:x86_64) target=x86_64-unknown-linux-musl ;;
  *) echo "Näky v0.1.0 supports Linux x86_64 only" >&2; exit 1 ;;
esac

release=${version#v}
archive="naky-${release}-${target}.tar.gz"
checksums="naky-${release}-SHA256SUMS"
base="https://github.com/${repository}/releases/download/${version}"
temporary=$(mktemp -d "${TMPDIR:-/tmp}/naky-install.XXXXXXXX")
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
  --output "$temporary/$archive" "$base/$archive"
curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error \
  --output "$temporary/$checksums" "$base/$checksums"

checksum=$(awk -v name="$archive" '$2 == name { print $1 }' "$temporary/$checksums")
case "$checksum" in
  [0-9a-f][0-9a-f]*) ;;
  *) echo "release checksum is missing" >&2; exit 1 ;;
esac
if [ "${#checksum}" -ne 64 ]; then
  echo "release checksum is malformed" >&2
  exit 1
fi
printf '%s  %s\n' "$checksum" "$temporary/$archive" | sha256sum --check --status

root="naky-${release}-${target}"
tar -tzf "$temporary/$archive" >"$temporary/members"
while IFS= read -r member; do
  case "$member" in
    "$root"|"$root/"|"$root/"*) ;;
    *) echo "unsafe archive member: $member" >&2; exit 1 ;;
  esac
  case "/$member/" in
    *'/../'*|*'/./'*) echo "unsafe archive member: $member" >&2; exit 1 ;;
  esac
done <"$temporary/members"
tar -tvzf "$temporary/$archive" >"$temporary/member-details"
while IFS= read -r line; do
  type=$(printf '%s' "$line" | cut -c1)
  case "$type" in
    -|d) ;;
    *) echo "archive contains a link or special file" >&2; exit 1 ;;
  esac
done <"$temporary/member-details"

tar -xzf "$temporary/$archive" -C "$temporary"
stage="$temporary/$root"
test -x "$stage/bin/naky"
test -d "$stage/share/naky/model"

install -d "$prefix/bin" "$prefix/share/naky"
install -m 0755 "$stage/bin/naky" "$prefix/bin/naky"
rm -rf "$prefix/share/naky/model"
install -d "$prefix/share/naky/model"
cp -R "$stage/share/naky/model/." "$prefix/share/naky/model/"
find "$prefix/share/naky/model" -type f -exec chmod 0644 {} +

"$prefix/bin/naky" --version
printf 'Installed Näky %s in %s\n' "$version" "$prefix"
