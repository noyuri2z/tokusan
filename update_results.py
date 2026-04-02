import sys

with open('tokusan/results.py', 'r') as f:
    content = f.read()

content = content.replace(
'''                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: precision={precision:.2f}, "
                    f"recall={recall:.2f}, f1={f1:.2f}"
                )

        return "\\n".join(lines)''', 
'''                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: precision={precision:.2f}, "
                    f"recall={recall:.2f}, f1={f1:.2f}"
                )

        if self.accuracy < 0.5:
            lines.append("")
            lines.append("正解率が50%以下です。モデルを変えるかデータセットに問題がないかを確認してください")

        return "\\n".join(lines)''')


content = content.replace(
'''                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: 適合率={precision:.2f}, "
                    f"再現率={recall:.2f}, F1={f1:.2f}"
                )

        return "\\n".join(lines)''',
'''                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: 適合率={precision:.2f}, "
                    f"再現率={recall:.2f}, F1={f1:.2f}"
                )

        if self.accuracy < 0.5:
            lines.append("")
            lines.append("正解率が50%以下です。モデルを変えるかデータセットに問題がないかを確認してください")

        return "\\n".join(lines)''')

with open('tokusan/results.py', 'w') as f:
    f.write(content)

