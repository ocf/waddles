import asyncio
from unittest.mock import patch

from llama_index.core.tools import FunctionTool
from ocflib.account.search import user_attrs
from ocflib.printing.quota import get_connection, get_quota

def create_user_info_tool() -> FunctionTool:
    """Create a tool to retrieve OCF user information and printing quotas."""

    def sync_fetch_quota(username: str):
        """Synchronous wrapper for fetching quota with a mocked group check."""
        with get_connection() as c:
            with patch('ocflib.printing.quota.is_in_group', return_value=False):
                return get_quota(c, username)

    async def get_ocf_user_info(username: str) -> str:
        """
        Get details about an OCF user, including their account creation time,
        UID, GID, email, and current printing quota.

        Args:
            username: The OCF username to look up.
        """
        try:
            # 1. Fetch User Attributes (Offloaded to thread)
            try:
                attrs = await asyncio.to_thread(user_attrs, username)
            except Exception as e:
                return f"Error finding user '{username}': {e}"

            if not attrs:
                return f"User '{username}' not found."

            # 2. Fetch Printing Quota (Offloaded to thread)
            quota_str = "Quota unavailable"
            try:
                quota = await asyncio.to_thread(sync_fetch_quota, username)
                if quota:
                    quota_str = (
                        f"Daily remaining: {quota.daily}, "
                        f"Semesterly remaining: {quota.semesterly}, "
                        f"Color remaining: {quota.color}"
                    )
            except Exception as e:
                quota_str = f"Error fetching quota: {e}"

            # 3. Format Output
            # LDAP values can sometimes be lists, so we safely unpack strings
            cn = attrs.get('cn', ['Unknown'])[0] if isinstance(attrs.get('cn'), list) else attrs.get('cn', 'Unknown')
            creation_time = attrs.get('creationTime', 'Unknown')

            output = [
                f"**User Info for:** {username}",
                f"- **Name:** {cn}",
                f"- **UID Number:** {attrs.get('uidNumber', 'Unknown')}",
                f"- **Email:** {attrs.get('ocfEmail', 'Unknown')}",
                f"- **Created:** {creation_time}",
                f"- **Printing Quota:** {quota_str}"
            ]
            return "\n".join(output)

        except Exception as e:
            return f"Tool execution failed: {e}"

    return FunctionTool.from_defaults(
        async_fn=get_ocf_user_info,
        name="get_ocf_user_info",
        description="Retrieve OCF user details (name, email, creation time) and printing quota limits by username. Useful when asked about a specific user's status or print limits."
    )