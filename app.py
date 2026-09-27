"""Python 3.10+ / Streamlit 1.64+。起動: python -m streamlit run othello.py"""

import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from queue import Empty as QueueEmpty, Queue
from threading import Thread

import streamlit as st


# 盤面は a1, b1, ... h8 の順に並ぶ、変更不可の64要素タプル。
EMPTY, BLACK, WHITE = 0, 1, -1
SIZE = 8
AI_MODE, HUMAN_MODE = "AIと対戦する", "近くの人と対戦する"
LEVELS = ["初心者", "中級者", "上級者", "プロ", "神"]
COLORS = {BLACK: "黒", WHITE: "白"}
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1)]
WEIGHTS = (
    100, -25, 10, 5, 5, 10, -25, 100,
    -25, -50, -2, -2, -2, -2, -50, -25,
    10, -2, 2, 1, 1, 2, -2, 10,
    5, -2, 1, 0, 0, 1, -2, 5,
    5, -2, 1, 0, 0, 1, -2, 5,
    10, -2, 2, 1, 1, 2, -2, 10,
    -25, -50, -2, -2, -2, -2, -50, -25,
    100, -25, 10, 5, 5, 10, -25, 100,
)
CORNER_NEIGHBORS = {
    0: (1, 8, 9), 7: (6, 14, 15),
    56: (48, 49, 57), 63: (54, 55, 62),
}


def build_rays():
    rays = []
    for pos in range(64):
        r, c = divmod(pos, SIZE)
        lines = []
        for dr, dc in DIRECTIONS:
            line = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < SIZE and 0 <= nc < SIZE:
                line.append(nr * SIZE + nc)
                nr, nc = nr + dr, nc + dc
            lines.append(tuple(line))
        rays.append(tuple(lines))
    return tuple(rays)


RAYS = build_rays()
NEIGHBORS = tuple(tuple(line[0] for line in rays if line) for rays in RAYS)


def initial_board():
    board = [EMPTY] * 64
    board[27] = board[36] = WHITE
    board[28] = board[35] = BLACK
    return tuple(board)


def coordinate(pos):
    return "pass" if pos is None else f"{'abcdefgh'[pos % 8]}{pos // 8 + 1}"


def flippable(board, pos, player):
    if not isinstance(pos, int) or not 0 <= pos < 64:
        return ()
    if player not in (BLACK, WHITE) or board[pos] != EMPTY:
        return ()
    flips = []
    for ray in RAYS[pos]:
        line = []
        for target in ray:
            if board[target] == -player:
                line.append(target)
            else:
                if board[target] == player and line:
                    flips.extend(line)
                break
    return tuple(flips)


def valid_moves(board, player):
    moves = {}
    for pos, value in enumerate(board):
        if value == EMPTY:
            flips = flippable(board, pos, player)
            if flips:
                moves[pos] = flips
    return moves


def put_disc(board, pos, player):
    flips = flippable(board, pos, player)
    if not flips:
        raise ValueError("そのマスには置けません。")
    result = list(board)
    result[pos] = player
    for target in flips:
        result[target] = player
    return tuple(result)


def next_turn(board, player):
    """戻り値: 次の手番、終局か、パスした色（なければNone）。"""
    if valid_moves(board, player):
        return player, False, None
    if valid_moves(board, -player):
        return -player, False, player
    return player, True, None


def final_score(board, player):
    diff = sum(board) * player
    if diff == 0:
        return 0
    return (100000 if diff > 0 else -100000) + diff * 100


def evaluate(board, player, my_moves=None, enemy_moves=None):
    my_moves = valid_moves(board, player) if my_moves is None else my_moves
    enemy_moves = valid_moves(board, -player) if enemy_moves is None else enemy_moves
    if not my_moves and not enemy_moves:
        return final_score(board, player)
    weights = list(WEIGHTS)
    for corner, neighbors in CORNER_NEIGHBORS.items():
        if board[corner] != EMPTY:
            for pos in neighbors:
                weights[pos] = 5
    positional = sum(value * weights[pos] for pos, value in enumerate(board)) * player
    mobility = len(my_moves) - len(enemy_moves)
    frontier = sum(
        value * player
        for pos, value in enumerate(board)
        if value and any(board[n] == EMPTY for n in NEIGHBORS[pos])
    )
    disc_weight = 1 if board.count(EMPTY) > 16 else 8
    return positional + mobility * 15 - frontier * 8 + sum(board) * player * disc_weight


class SearchTimeout(Exception):
    pass


def builtin_ai(board, player, level):
    """時間制限付き反復深化・αβ探索。世界最高水準を称するAIではない。"""
    moves = valid_moves(board, player)
    if not moves:
        return None, "パス"
    if level == "初心者":
        return random.choice(list(moves)), "ランダム"
    max_depth, seconds = {
        "中級者": (2, 0.3), "上級者": (4, 1.0), "プロ": (12, 3.0),
    }[level]
    deadline = time.perf_counter() + seconds
    table = {}
    nodes = 0

    def search(position, color, depth, alpha, beta):
        nonlocal nodes
        nodes += 1
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        key = (position, color)
        alpha_start, beta_start = alpha, beta
        entry = table.get(key)
        preferred = entry[3] if entry else None
        if entry and entry[0] >= depth:
            _, value, bound, move = entry
            if bound == "exact":
                return value, move
            if bound == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value, move
        legal = valid_moves(position, color)
        if not legal:
            if not valid_moves(position, -color):
                return final_score(position, color), None
            value, _ = search(position, -color, depth, -beta, -alpha)
            return -value, None  # パスでは残り探索手数を減らさない。
        if depth == 0:
            return evaluate(position, color, legal), None
        ordered = sorted(legal, key=lambda p: (p == preferred, WEIGHTS[p]), reverse=True)
        best_value, best_move = -float("inf"), ordered[0]
        for move in ordered:
            child = put_disc(position, move, color)
            value, _ = search(child, -color, depth - 1, -beta, -alpha)
            value = -value
            if value > best_value:
                best_value, best_move = value, move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        bound = "upper" if best_value <= alpha_start else (
            "lower" if best_value >= beta_start else "exact"
        )
        table[key] = (depth, best_value, bound, best_move)
        return best_value, best_move

    best_move = max(moves, key=lambda p: evaluate(put_disc(board, p, player), player))
    completed = 0
    for depth in range(1, min(max_depth, board.count(EMPTY)) + 1):
        try:
            _, move = search(board, player, depth, -float("inf"), float("inf"))
        except SearchTimeout:
            break
        if move is not None:
            best_move, completed = move, depth
    detail = "終局まで探索" if completed == board.count(EMPTY) else f"{completed}手先まで探索完了"
    return best_move, f"{detail} / {nodes:,}局面"


# Edax 4.6のGTPを使用。外部実行ファイルはユーザーが指定する。
class EngineError(RuntimeError):
    pass


def checked_engine_paths(executable, evaluation, book=""):
    def path_of(text):
        return Path(text.strip().strip('"')).expanduser().resolve()

    if not executable.strip():
        raise EngineError("神にはEdaxが必要です。実行ファイルのパスを設定してください。")
    exe = path_of(executable)
    if not exe.is_file():
        raise EngineError("Edaxの実行ファイルが見つかりません。")
    if os.name != "nt" and not os.access(exe, os.X_OK):
        raise EngineError("Edaxに実行権限がありません。実行権限を設定してください。")
    candidates = [exe.parent / "data" / "eval.dat", exe.parent / "eval.dat",
                  exe.parent.parent / "data" / "eval.dat"]
    weight = path_of(evaluation) if evaluation.strip() else next(
        (p for p in candidates if p.is_file()), None
    )
    if weight is None or not weight.is_file():
        raise EngineError("評価データeval.datが見つかりません。パスを指定してください。")
    book_path = path_of(book) if book.strip() else None
    if book_path is not None and not book_path.is_file():
        raise EngineError("指定された定石ファイルが見つかりません。")
    return exe, weight, book_path


class EdaxClient:
    def __init__(self, executable, evaluation, book="", seconds=10):
        exe, weight, book_path = checked_engine_paths(executable, evaluation, book)
        self.scratch = tempfile.TemporaryDirectory(prefix="othello_edax_")
        local_book = Path(self.scratch.name) / "book.dat"
        if book_path:
            shutil.copyfile(book_path, local_book)
        command = [
            str(exe), "-gtp", "-eval-file", str(weight),
            # 未作成の定石データの初期化だけ低レベルで行う。
            # 実際の思考はGTPのtime_leftでレベル60に設定される。
            "-level", "1", "-hash-table-size", "22",
            "-n-tasks", str(max(1, min(8, os.cpu_count() or 1))),
            "-ponder", "off", "-auto-store", "off", "-book-randomness", "0",
            "-book-usage", "on" if book_path else "off",
            "-book-file", str(local_book),
        ]
        try:
            self.process = subprocess.Popen(
                command, cwd=self.scratch.name, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self.scratch.cleanup()
            raise
        self.lines = Queue()
        self.command_id = 0
        self.seconds = seconds
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line.rstrip("\r\n"))
        finally:
            self.lines.put(None)

    def ask(self, command, timeout=30):
        self.command_id += 1
        request_id = self.command_id
        try:
            self.process.stdin.write(f"{request_id} {command}\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError("Edaxへの送信に失敗しました。") from exc
        deadline = time.monotonic() + timeout
        status, response, diagnostics = None, [], []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("Edaxの応答が時間切れになりました。")
            try:
                line = self.lines.get(timeout=remaining)
            except QueueEmpty as exc:
                raise EngineError("Edaxの応答が時間切れになりました。") from exc
            if line is None:
                raise EngineError("Edaxが終了しました。 " + " / ".join(diagnostics[-3:]))
            if status is None:
                match = re.match(r"^([=?])\s*(\d+)\s*(.*)$", line)
                if match and int(match.group(2)) == request_id:
                    status = match.group(1)
                    response.append(match.group(3))
                else:
                    diagnostics.append(line)
            elif not line.strip():
                text = "\n".join(response).strip()
                if status == "?":
                    raise EngineError(f"Edaxがコマンドを拒否しました: {command}: {text}")
                return text
            else:
                response.append(line)

    def close(self):
        if self.process.poll() is None:
            try:
                self.ask("quit", timeout=2)
                self.process.wait(timeout=2)
            except (EngineError, OSError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
        self.scratch.cleanup()


def edax_ai(board, player, history, config):
    client = None
    try:
        client = EdaxClient(config["exe"], config["eval"], config["book"], config["seconds"])
        if client.ask("name").strip().lower() != "edax":
            raise EngineError("指定されたエンジンはEdaxではありません。")
        version = client.ask("version")
        client.ask("clear_board")
        for color, move in history:
            token = "black" if color == BLACK else "white"
            client.ask(f"play {token} {coordinate(move)}")
        token = "black" if player == BLACK else "white"
        seconds = int(config["seconds"])
        client.ask(f"time_settings {seconds} 0 1")
        client.ask(f"time_left {token} {seconds} 1")
        answer = client.ask(f"genmove {token}", timeout=seconds + 30).lower()
        if not re.fullmatch(r"[a-h][1-8]", answer):
            raise EngineError(f"Edaxの着手を解釈できません: {answer}")
        move = (int(answer[1]) - 1) * 8 + ord(answer[0]) - ord("a")
        if move not in valid_moves(board, player):
            raise EngineError("Edaxが合法でない着手を返しました。")
        return move, f"Edax {version} / 思考時間設定 {seconds}秒"
    except OSError as exc:
        raise EngineError(f"Edaxを実行できません。OS・CPU・パスを確認してください: {exc}") from exc
    finally:
        if client is not None:
            client.close()


def new_game():
    s = st.session_state
    s.board, s.turn, s.game_over = initial_board(), BLACK, False
    s.history, s.pass_notice, s.ai_info, s.ai_error = [], "", "", ""
    s.game_number = s.get("game_number", 0) + 1


def commit_move(move):
    s = st.session_state
    color = s.turn
    s.board = put_disc(s.board, move, color)
    s.history.append((color, move))
    s.turn, s.game_over, passed = next_turn(s.board, -color)
    if passed is not None:
        s.history.append((passed, None))
        s.pass_notice = f"{COLORS[passed]}は置ける場所がないためパスしました。"


def human_move(move, expected_board, expected_turn):
    s = st.session_state
    if s.game_over or s.board != expected_board or s.turn != expected_turn:
        return
    if s.config["mode"] == AI_MODE and s.turn != BLACK:
        return
    if move in valid_moves(s.board, s.turn):
        s.pass_notice, s.ai_error = "", ""
        commit_move(move)


BOARD_CSS = """
<style>
.st-key-reversi_board [data-testid="stVerticalBlock"] {gap:0 !important;}
.st-key-reversi_board [data-testid="stHorizontalBlock"] {gap:0 !important;}
.st-key-reversi_board [data-testid="stColumn"] {min-width:0 !important; padding:0 !important;}
.st-key-reversi_board [data-testid="stButton"] {width:100% !important;}
.st-key-reversi_board button {
    background:#16843b !important; border:1px solid #094b22 !important;
    border-radius:0 !important; width:100% !important; aspect-ratio:1 / 1;
    min-height:0 !important; height:auto !important; padding:0 !important;
    margin:0 !important; opacity:1 !important; color:white !important;
    box-shadow:none !important; position:relative;
}
.st-key-reversi_board button p {
    font-size:0 !important; line-height:0 !important; margin:0 !important;
}
.st-key-reversi_board button::after {
    position:absolute; left:50%; top:50%; transform:translate(-50%, -50%);
    width:70%; aspect-ratio:1 / 1; border-radius:50%; pointer-events:none;
}
.st-key-reversi_board button:enabled:hover {background:#24ac52 !important;}
.st-key-reversi_board button:enabled {cursor:pointer;}
.st-key-reversi_board button:disabled {cursor:default;}
.st-key-reversi_board [data-testid="stMarkdown"] {
    min-height:28px; text-align:center;
}
.st-key-reversi_board [data-testid="stMarkdown"] p {
    margin:0 !important; line-height:28px;
}
</style>
"""


def main():
    st.set_page_config(page_title="オセロ（リバーシ）", page_icon="🟢", layout="centered")
    s = st.session_state
    if s.get("reversi_version") != 1:
        s.reversi_version = 1
        s.config = {"mode": AI_MODE, "level": "中級者", "exe": os.getenv("EDAX_PATH", ""),
                    "eval": os.getenv("EDAX_EVAL", ""), "book": "", "seconds": 10}
        new_game()

    with st.sidebar:
        st.header("対戦設定")
        st.caption("変更後は「この設定で新しく開始」を押してください。")
        with st.form("settings"):
            mode = st.radio("対戦モード", [AI_MODE, HUMAN_MODE], key="setting_mode")
            level = st.selectbox("AIのレベル", LEVELS, index=1, key="setting_level")
            st.caption("神はEdaxを使用します。その他のレベルは内蔵AIです。")
            exe = st.text_input("Edax実行ファイルのパス", value=s.config["exe"])
            evaluation = st.text_input("eval.datのパス（自動検出できれば空欄可）", value=s.config["eval"])
            book = st.text_input("Edax定石ファイルのパス（任意）", value=s.config["book"])
            seconds = st.slider("神の1手の思考時間（秒）", 1, 60, 10)
            start = st.form_submit_button("この設定で新しく開始", type="primary")
        if start:
            try:
                if mode == AI_MODE and level == "神":
                    checked_engine_paths(exe, evaluation, book)
            except (EngineError, OSError) as exc:
                st.error(str(exc))
            else:
                s.config = {"mode": mode, "level": level, "exe": exe,
                            "eval": evaluation, "book": book, "seconds": seconds}
                new_game()
                st.rerun()
        st.markdown("[Edax公式ダウンロード](https://github.com/abulmo/edax-reversi/releases)")
        st.caption("Edax 4.6のGTPに対応。OS・CPUに合う実行ファイルと評価データが必要です。")

    config = dict(s.config)
    st.title("🟢 オセロゲーム")
    active = config["mode"]
    if active == AI_MODE:
        active += f" / {config['level']} / あなたは黒、AIは白"
    st.caption(f"現在の対戦：{active}")
    st.button("ゲームをやり直す", type="primary", on_click=new_game)
    black, white = s.board.count(BLACK), s.board.count(WHITE)
    st.subheader(f"⚫ 黒：{black}　｜　⚪ 白：{white}")
    if s.game_over:
        winner = "引き分け！" if black == white else f"{'黒' if black > white else '白'}の勝利！"
        st.success(f"対局終了：{winner}")
    else:
        role = ""
        if config["mode"] == AI_MODE:
            role = "（あなた）" if s.turn == BLACK else "（AI）"
        st.info(f"現在の手番：{COLORS[s.turn]}{role}")
    if s.pass_notice:
        st.warning(s.pass_notice)
    if s.ai_info:
        st.caption(f"直前のAI：{s.ai_info}")

    legal = valid_moves(s.board, s.turn)
    human_turn = not s.game_over and (config["mode"] == HUMAN_MODE or s.turn == BLACK)
    disc_css = []
    for pos, value in enumerate(s.board):
        selector = f".st-key-reversi_board .st-key-cell_{s.game_number}_{pos} button::after"
        if value:
            color = "#111111" if value == BLACK else "#ffffff"
            style = f'content:""; background:{color}; border:1px solid #555;'
        elif human_turn and pos in legal:
            style = 'content:""; width:16%; background:#c0eccb;'
        else:
            continue
        disc_css.append(f"{selector} {{{style}}}")
    st.markdown(BOARD_CSS + "<style>" + "\n".join(disc_css) + "</style>", unsafe_allow_html=True)
    with st.container(key="reversi_board", gap=None):
        header = st.columns([0.45] + [1] * 8, gap=None, wrap=False)
        for col, letter in zip(header[1:], "abcdefgh"):
            col.markdown(f"**{letter}**")
        for r in range(SIZE):
            columns = st.columns([0.45] + [1] * 8, gap=None, vertical_alignment="center", wrap=False)
            columns[0].markdown(f"**{r + 1}**")
            for c in range(SIZE):
                pos = r * 8 + c
                value = s.board[pos]
                allowed = human_turn and pos in legal
                label = f"{coordinate(pos)} {COLORS.get(value, '置ける場所' if allowed else '空き')}"
                columns[c + 1].button(
                    label, key=f"cell_{s.game_number}_{pos}", width="stretch",
                    disabled=not allowed, help=f"{coordinate(pos)}：{COLORS.get(value, '空き')}",
                    on_click=human_move, args=(pos, s.board, s.turn),
                )
    st.caption("「・」のマスに置けます。置ける場所がないときは自動でパスします。")
    if s.history:
        recent = " → ".join(f"{COLORS[p]} {coordinate(m)}" for p, m in s.history[-4:])
        st.caption(f"直近の手順：{recent}")
        with st.expander("すべての棋譜"):
            st.text("\n".join(f"{i + 1:2d}. {COLORS[p]} {coordinate(m)}"
                              for i, (p, m) in enumerate(s.history)))

    if s.game_over or config["mode"] != AI_MODE or s.turn != WHITE:
        return
    if s.ai_error:
        st.error(s.ai_error)
        if st.button("AIの着手を再試行"):
            s.ai_error = ""
            st.rerun()
        return
    expected_board = s.board
    try:
        with st.spinner(f"{config['level']}のAIが思考中…"):
            if config["level"] == "神":
                move, info = edax_ai(s.board, WHITE, tuple(s.history), config)
            else:
                move, info = builtin_ai(s.board, WHITE, config["level"])
        if s.board == expected_board and s.turn == WHITE and s.config == config:
            if move not in valid_moves(s.board, WHITE):
                raise EngineError("AIから合法な着手を取得できませんでした。")
            commit_move(move)
            s.ai_info = info
            st.rerun()
    except EngineError as exc:
        s.ai_error = str(exc)
        st.rerun()


if __name__ == "__main__":
    main()
