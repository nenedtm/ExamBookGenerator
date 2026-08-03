**Date:** {{date:DD/MM/YYYY}}

---

> [!recap] 🟡 Lecture Recap
> 
> ```dataviewjs
> const content = await dv.io.load(dv.current().file.path);
> const lines = content.split("\n");
> const headings = [];
> 
> for (const line of lines) {
>   const match = line.match(/^(#{2,4})\s+(.+)/);
>   if (match) {
>     const level = match[1].length;
>     const title = match[2].trim();
>     const indent = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".repeat(level - 2);
>     const prefix = level === 2 ? "▸ " : level === 3 ? "• " : "◦ ";
>     const link = dv.fileLink(dv.current().file.path + "#" + title, false, title);
>     dv.paragraph(indent + prefix + link);
>   }
> }
> ```

---

# 📝 Notes





















---

## 🔗 References & Resources

---

> [!tip]- How to use this template
> 
> |Element|Callout syntax|
> |---|---|
> |🟡 Recap|`> [!recap] Title`|
> |🟢 Formula|`> [!formula] Formula Name`|
> |🔵 Theorem|`> [!theorem] Theorem Name`|
> |🟣 Proof|`> [!proof] Proof — Name`|
> |🟠 Question|`> [!question] Titolo`|
> |🩵 Exercise|`> [!exercise] Titolo`|
> |🔴 Review|`> [!review] Note`|
> 
> All callouts fully support LaTeX — just write `$$...$$` inside them normally.  
> ⚙️ Enable the CSS snippet: _Settings → Appearance → CSS Snippets → university-notes_
> 
> **Auto-index:** Requires the **Dataview** plugin with _Enable JavaScript Queries_ turned on in its settings. The recap updates live as you add `##`, `###`, `####` headings.