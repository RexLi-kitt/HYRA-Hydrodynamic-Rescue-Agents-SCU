"""Record Java cache access from the user's interactive Windows session."""
import os
from pathlib import Path
import subprocess
import sys
import ctypes
from ctypes import wintypes

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'freeterraforged-reproduction' / '.runtime'


def main():
    cache = Path(os.environ['GRADLE_USER_HOME'])
    log = ROOT / '.runtime' / 'windows-cache-check.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'Python: {sys.executable}', f'Cache: {cache}']
    # A normal user name does not establish whether the process token is restricted.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    security = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    security.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    security.IsTokenRestricted.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    token = wintypes.HANDLE()
    if security.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
        try:
            restricted = bool(security.IsTokenRestricted(token))
            lines.append(f'Process token restricted: {restricted}')
            print(f'Process token restricted: {restricted}')
        finally:
            kernel.CloseHandle(token)
    identity = subprocess.run(['whoami', '/user'], capture_output=True)
    lines.append(identity.stdout.decode(errors='replace'))
    groups = subprocess.run(['whoami', '/groups'], capture_output=True)
    lines.append(groups.stdout.decode('gbk', errors='replace'))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / 'hyra-write-check.txt'
        probe.write_text('HYRA write check\n', encoding='utf-8')
        lines.append('Python cache write: OK')
        java = next(RUNTIME.glob('java/*/bin/java.exe'))
        source = ROOT / '.runtime' / 'CacheAccessCheck.java'
        source.write_text('''import java.nio.file.*;
import java.util.*;
public class CacheAccessCheck {
 public static void main(String[] args) throws Exception {
  Path dir = Path.of(args[0]);
  Path archive = dir.resolve("hyra-java-write-check-" + UUID.randomUUID() + ".jar");
  System.out.println("Java user: " + System.getProperty("user.name"));
  Process who = new ProcessBuilder("whoami", "/groups").redirectErrorStream(true).start();
  who.getInputStream().transferTo(System.out);
  who.waitFor();
  System.out.println("Directory writable: " + Files.isWritable(dir));
  Path plain = dir.resolve("hyra-java-plain-" + UUID.randomUUID() + ".txt");
  try {
   Files.writeString(plain, "HYRA Java write check");
   System.out.println("Java plain file write: OK");
  } catch (Exception e) {
   System.out.println("Java plain file write: " + e);
  } finally { Files.deleteIfExists(plain); }
  try (var fs = FileSystems.newFileSystem(archive, Map.of("create", "true"))) {
   System.out.println("JAR readOnly: " + fs.isReadOnly());
   Files.createDirectories(fs.getPath("/hyra"));
   System.out.println("Java JAR write: OK");
  } finally { Files.deleteIfExists(archive); }
 }
}
''', encoding='utf-8')
        result = subprocess.run([str(java), '-Dfile.encoding=UTF-8', str(source), str(cache)],
                                capture_output=True, timeout=45)
        lines.extend([result.stdout.decode('utf-8', errors='replace'),
                      result.stderr.decode('utf-8', errors='replace')])
        code = result.returncode
    except Exception as error:
        lines.append(f'{type(error).__name__}: {error}')
        code = 1
    log.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Cache check: {"OK" if code == 0 else "FAILED"}')
    print(f'Diagnostic log: {log}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
