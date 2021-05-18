<script lang="ts">
	import { actual_user,users } from './store.js';

	import Menu from '@smui/menu';
	import List, { Item, Separator, Text } from '@smui/list';
	import Button from '@smui/button';
	let users_list = []
	let actual;
	users.subscribe(value => {
		users_list = value;
	});
	actual_user.subscribe(value =>{
		actual = value
	})
	let menu;
</script>

<pre class="status">Użytkownik: {actual.name}</pre>
	<div style="min-width: 100px;">
		<Button style="color:red;"on:click={() => menu.setOpen(true)}>User Menu</Button>
		<Menu bind:this={menu}>
		  <List>
			{#each users_list as user}
			<Item on:SMUI:action={() => (actual_user.set(user))}>
			  <Text>{user.name}</Text>
			</Item>
			{/each}
			<Separator />
		  </List>
		</Menu>
	  </div>
